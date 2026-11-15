from __future__ import annotations

from datetime import date
from typing import Any, Dict

import pandas as pd
import psutil
import streamlit as st

from config import (
    ALL_CATEGORIES,
    DEFAULT_CURRENCY_LABEL,
    SECURITY_QUESTIONS,
    SUPPORTED_CURRENCIES,
)
from database.db import (
    authenticate_user,
    clear_all_transactions,
    commit_verified_transactions,
    configure_or_rotate_recovery_credentials,
    create_user,
    delete_user_account,
    export_user_backup_bytes,
    get_all_corrections,
    get_forecasts_dataframe,
    get_transactions_dataframe,
    get_user_recovery_profile,
    init_db,
    reset_user_password_offline,
    restore_user_backup_bytes,
    sync_ledger_crud,
    update_user_currency,
)
from ingestion.categorizer import HybridCategorizer
from ingestion.pdf_extractor import StatementExtractor
from llm_engine.hf_manager import LocalHFEngine
from mcp_investment.advisor_client import MCPInvestmentOrchestrator
from rl_engine.insights import compute_spend_analytics, generate_ai_spend_commentary
from rl_engine.trainer import RLForecastManager

st.set_page_config(
    page_title="LocalFinAI — Private Personal Finance & Investment Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.16);
        padding: 14px 18px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()
hf_engine = LocalHFEngine()


def get_process_memory_mb() -> float:
    process = psutil.Process()
    return round(process.memory_info().rss / (1024 * 1024), 1)


def clear_user_session_cache() -> None:
    for key in (
        "staging_df",
        "initial_ai_categories",
        "latest_rl_pred_df",
        "ai_commentary_text",
        "mcp_advisor_result",
    ):
        st.session_state.pop(key, None)


def build_recovery_kit_text(username: str, display_name: str, recovery_key: str, question: str) -> str:
    return (
        "============================================================\n"
        "        LOCALFINAI — EMERGENCY ACCOUNT RECOVERY KIT         \n"
        "============================================================\n\n"
        f"Account Holder : {display_name}\n"
        f"Username       : {username}\n"
        f"Recovery Key   : {recovery_key}\n"
        f"Security Q.    : {question}\n\n"
        "IMPORTANT:\n"
        "- LocalFinAI is 100% offline. Your credentials never leave this device.\n"
        "- Keep this Recovery Key in a safe place. You can use either this\n"
        "  Recovery Key OR your Security Question Answer to reset a forgotten\n"
        "  password on the 'Forgot Password' tab.\n"
        "============================================================\n"
    )


# ============================================================================
# Authentication Gate: Sign In / Create Account / Forgot Password
# ============================================================================
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

if st.session_state["auth_user"] is None:
    _, center_col, _ = st.columns([1.1, 1.8, 1.1])
    with center_col:
        st.markdown("## 🛡️ LocalFinAI")
        st.caption(
            "100% Local & Private Financial Intelligence • Isolated Multi-User Workspaces"
        )

        deleted_summary = st.session_state.pop("deleted_account_summary", None)
        if deleted_summary:
            st.success(
                f"Account `@{deleted_summary['username']}` and all associated data "
                f"({deleted_summary['deleted_transactions']} transactions, "
                f"{deleted_summary['deleted_rules']} learned rules, and "
                f"{deleted_summary['deleted_forecasts']} forecasts) have been permanently deleted."
            )

        auth_tab_login, auth_tab_signup, auth_tab_recover = st.tabs(
            ["Sign In", "Create New Account", "Forgot Password"]
        )

        # --------------------------------------------------------------------
        # 1. Sign In Tab
        # --------------------------------------------------------------------
        with auth_tab_login:
            with st.form("login_form", clear_on_submit=False):
                login_username = st.text_input("Username", placeholder="e.g., ayush")
                login_password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button(
                    "Sign In to Workspace", type="primary", use_container_width=True
                )
                if submit_login:
                    user = authenticate_user(login_username, login_password)
                    if user is not None:
                        clear_user_session_cache()
                        st.session_state["auth_user"] = {
                            "id": int(user.id),  # type: ignore[arg-type]
                            "username": user.username,
                            "display_name": user.display_name,
                            "preferred_currency": user.preferred_currency,
                        }
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Use the 'Forgot Password' tab if you need to recover access.")

        # --------------------------------------------------------------------
        # 2. Create New Account Tab
        # --------------------------------------------------------------------
        with auth_tab_signup:
            st.caption(
                "New accounts start with a blank slate and include offline Dual-Factor Account Recovery (Master Recovery Key + Security Question)."
            )
            with st.form("signup_form", clear_on_submit=False):
                c_su1, c_su2 = st.columns(2)
                with c_su1:
                    new_display = st.text_input("Full Name / Display Name", placeholder="e.g., Ayush Sharma")
                with c_su2:
                    new_username = st.text_input("Choose Username (no spaces)", placeholder="e.g., ayush")

                new_currency = st.selectbox(
                    "Default Currency",
                    options=list(SUPPORTED_CURRENCIES.keys()),
                    index=list(SUPPORTED_CURRENCIES.keys()).index(DEFAULT_CURRENCY_LABEL),
                )

                c_pw1, c_pw2 = st.columns(2)
                with c_pw1:
                    new_password = st.text_input("Create Password (min. 6 chars)", type="password")
                with c_pw2:
                    new_password_confirm = st.text_input("Confirm Password", type="password")

                st.markdown("###### 🔐 Account Recovery Setup (100% Offline Verification)")
                sec_question = st.selectbox("Select a Security Question", options=SECURITY_QUESTIONS)
                sec_answer = st.text_input(
                    "Your Security Answer (case-insensitive)",
                    placeholder="Enter a memorable answer...",
                )

                submit_signup = st.form_submit_button(
                    "Create Blank Account & Generate Recovery Key",
                    type="primary",
                    use_container_width=True,
                )

                if submit_signup:
                    if new_password != new_password_confirm:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            created_user, plaintext_recovery_key = create_user(
                                username=new_username,
                                password=new_password,
                                display_name=new_display,
                                preferred_currency=new_currency,
                                security_question=sec_question,
                                security_answer=sec_answer,
                            )
                            clear_user_session_cache()
                            st.session_state["auth_user"] = {
                                "id": int(created_user.id),  # type: ignore[arg-type]
                                "username": created_user.username,
                                "display_name": created_user.display_name,
                                "preferred_currency": created_user.preferred_currency,
                            }
                            st.session_state["new_recovery_kit"] = {
                                "username": created_user.username,
                                "display_name": created_user.display_name,
                                "recovery_key": plaintext_recovery_key,
                                "security_question": sec_question,
                            }
                            st.rerun()
                        except ValueError as err:
                            st.error(str(err))

        # --------------------------------------------------------------------
        # 3. Forgot Password / Offline Account Recovery Tab
        # --------------------------------------------------------------------
        with auth_tab_recover:
            st.caption(
                "Recover your account 100% offline using either your **Security Question Answer** or your **Master Recovery Key (`LFAI-XXXX-XXXX-XXXX`)**."
            )
            lookup_col1, lookup_col2 = st.columns([2.2, 1.0])
            with lookup_col1:
                rec_username_input = st.text_input(
                    "Enter your Username",
                    value=st.session_state.get("recovery_target_user", ""),
                    placeholder="e.g., ayush",
                    key="rec_user_lookup_input",
                )
            with lookup_col2:
                st.write("")
                if st.button("Find Account", use_container_width=True):
                    profile = get_user_recovery_profile(rec_username_input)
                    if profile is None:
                        st.session_state.pop("recovery_profile", None)
                        st.error("No account found with that username.")
                    else:
                        st.session_state["recovery_target_user"] = profile["username"]
                        st.session_state["recovery_profile"] = profile

            rec_profile = st.session_state.get("recovery_profile")
            if rec_profile:
                st.markdown(
                    f"**Account Found:** `{rec_profile['display_name']}` (`@{rec_profile['username']}`)"
                )

                if not rec_profile["has_security_question"] and not rec_profile["has_recovery_key"]:
                    st.warning(
                        "This legacy account was created before offline recovery was enabled. "
                        "Sign in and open '🔐 Account Recovery Settings' in the sidebar to configure a Recovery Key."
                    )
                else:
                    method_options = []
                    if rec_profile["has_security_question"]:
                        method_options.append("Answer Security Question")
                    if rec_profile["has_recovery_key"]:
                        method_options.append("Enter Master Recovery Key (LFAI-...)")

                    chosen_method_label = st.radio(
                        "Select Verification Method",
                        options=method_options,
                        horizontal=True,
                    )

                    with st.form("offline_password_reset_form", clear_on_submit=False):
                        if chosen_method_label == "Answer Security Question":
                            st.info(f"**Security Question:** {rec_profile['security_question']}")
                            verification_val = st.text_input(
                                "Your Answer (case-insensitive)",
                                type="password",
                                placeholder="Type your security answer...",
                            )
                            method_code = "security_question"
                        else:
                            verification_val = st.text_input(
                                "Master Recovery Key",
                                placeholder="LFAI-XXXX-XXXX-XXXX",
                            )
                            method_code = "recovery_key"

                        r_pw1, r_pw2 = st.columns(2)
                        with r_pw1:
                            reset_new_pw = st.text_input("New Password (min. 6 chars)", type="password")
                        with r_pw2:
                            reset_confirm_pw = st.text_input("Confirm New Password", type="password")

                        submit_reset = st.form_submit_button(
                            "Verify & Reset Password",
                            type="primary",
                            use_container_width=True,
                        )

                        if submit_reset:
                            if reset_new_pw != reset_confirm_pw:
                                st.error("New passwords do not match.")
                            else:
                                try:
                                    updated_user = reset_user_password_offline(
                                        username=rec_profile["username"],
                                        new_password=reset_new_pw,
                                        verification_method=method_code,
                                        verification_input=verification_val,
                                    )
                                    st.session_state.pop("recovery_profile", None)
                                    st.session_state.pop("recovery_target_user", None)
                                    clear_user_session_cache()
                                    st.session_state["auth_user"] = {
                                        "id": int(updated_user.id),  # type: ignore[arg-type]
                                        "username": updated_user.username,
                                        "display_name": updated_user.display_name,
                                        "preferred_currency": updated_user.preferred_currency,
                                    }
                                    st.session_state["pw_reset_success_banner"] = True
                                    st.rerun()
                                except ValueError as err:
                                    st.error(str(err))

    st.stop()


# ============================================================================
# Authenticated Workspace Context
# ============================================================================
current_user: Dict[str, Any] = st.session_state["auth_user"]
USER_ID: int = int(current_user["id"])

# Pre-build portable backup ZIP bytes for instant one-click download in sidebar
backup_zip_bytes = export_user_backup_bytes(USER_ID)
backup_filename = f"LocalFinAI_Backup_{current_user['username']}_{date.today().isoformat()}.zip"

# ============================================================================
# Minimal Sidebar: Profile, Currency, Backup/Restore, Recovery, AI & Delete
# ============================================================================
with st.sidebar:
    st.markdown(f"### 👤 {current_user['display_name']}")
    st.caption(f"Account: `@{current_user['username']}` • Offline SQLite")

    saved_curr = current_user.get("preferred_currency", DEFAULT_CURRENCY_LABEL)
    curr_keys = list(SUPPORTED_CURRENCIES.keys())
    curr_idx = curr_keys.index(saved_curr) if saved_curr in curr_keys else 0

    selected_currency_label = st.selectbox(
        "Workspace Currency",
        options=curr_keys,
        index=curr_idx,
    )
    if selected_currency_label != saved_curr:
        update_user_currency(USER_ID, selected_currency_label)
        st.session_state["auth_user"]["preferred_currency"] = selected_currency_label

    CURRENCY: str = SUPPORTED_CURRENCIES[selected_currency_label]

    corrections_map = get_all_corrections(USER_ID)
    st.caption(f"Learned Merchant Rules: **{len(corrections_map)}**")

    # ------------------------------------------------------------------------
    # One-Click Export & Restore Backup Panel
    # ------------------------------------------------------------------------
    with st.expander("📦 Backup & Restore Workspace", expanded=False):
        st.caption(
            "Export your ledger, learned merchant rules, RL forecasts, and trained PPO model into a single `.zip` archive, or restore from an existing backup."
        )
        st.download_button(
            "⬇️ Export Full Account Backup (.zip)",
            data=backup_zip_bytes,
            file_name=backup_filename,
            mime="application/zip",
            type="primary",
            use_container_width=True,
            key="sidebar_export_backup_btn",
        )
        st.divider()
        restore_file = st.file_uploader(
            "Restore Backup (.zip or .json)",
            type=["zip", "json"],
            key="sidebar_restore_uploader",
        )
        restore_mode = st.radio(
            "Import Mode",
            options=["Merge (Skip Duplicates)", "Replace Entire Workspace"],
            key="sidebar_restore_mode",
        )
        if st.button(
            "🔄 Restore Backup Now",
            disabled=restore_file is None,
            use_container_width=True,
        ):
            if restore_file is not None:
                try:
                    stats = restore_user_backup_bytes(
                        user_id=USER_ID,
                        file_bytes=restore_file.read(),
                        replace_existing=(restore_mode == "Replace Entire Workspace"),
                    )
                    if stats.get("preferred_currency"):
                        st.session_state["auth_user"]["preferred_currency"] = stats["preferred_currency"]
                    clear_user_session_cache()
                    st.session_state["restore_success_banner"] = stats
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))

    with st.expander("🔐 Account Recovery Settings", expanded=False):
        st.caption(
            "Update your Security Question or generate a new Master Recovery Key (`LFAI-...`)."
        )
        upd_q = st.selectbox(
            "Security Question",
            options=SECURITY_QUESTIONS,
            key="sidebar_sec_q",
        )
        upd_a = st.text_input(
            "New Security Answer",
            type="password",
            placeholder="Enter answer...",
            key="sidebar_sec_a",
        )
        if st.button("Generate New Recovery Kit", use_container_width=True):
            try:
                fresh_key = configure_or_rotate_recovery_credentials(
                    user_id=USER_ID,
                    security_question=upd_q,
                    security_answer=upd_a,
                )
                st.session_state["new_recovery_kit"] = {
                    "username": current_user["username"],
                    "display_name": current_user["display_name"],
                    "recovery_key": fresh_key,
                    "security_question": upd_q,
                }
                st.rerun()
            except ValueError as err:
                st.error(str(err))

    with st.expander("⚙️ Local AI & Memory Status", expanded=False):
        rss_mb = get_process_memory_mb()
        sys_mem = psutil.virtual_memory()
        st.metric(
            "App RAM",
            f"{rss_mb:,.0f} MB",
            f"{sys_mem.available / (1024**3):.1f} GB Free",
        )
        st.caption(
            f"Qwen2.5-3B Status: **{'Loaded' if hf_engine.is_llm_loaded() else 'Standby'}**"
        )
        c_l1, c_l2 = st.columns(2)
        with c_l1:
            if st.button("Load LLM", use_container_width=True):
                with st.spinner("Loading GGUF..."):
                    hf_engine.load_llm()
                st.rerun()
        with c_l2:
            if st.button("Free RAM", use_container_width=True):
                hf_engine.unload_llm()
                st.rerun()

    with st.expander("⚠️ Delete Account & Data", expanded=False):
        st.caption(
            "Permanently delete your account (`@"
            + str(current_user["username"])
            + "`), transactions, category rules, forecasts, and trained RL model."
        )
        st.download_button(
            "⬇️ Download Backup Before Deleting (.zip)",
            data=backup_zip_bytes,
            file_name=backup_filename,
            mime="application/zip",
            use_container_width=True,
            key="pre_delete_backup_btn",
        )
        del_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Enter your password...",
            key="del_acc_pw",
        )
        del_confirm_check = st.checkbox(
            "I understand this action is permanent and cannot be undone.",
            key="del_acc_chk",
        )
        if st.button(
            "🗑️ Permanently Delete Account",
            type="primary",
            disabled=not del_confirm_check,
            use_container_width=True,
        ):
            try:
                summary = delete_user_account(user_id=USER_ID, password=del_password)
                clear_user_session_cache()
                st.session_state.pop("new_recovery_kit", None)
                st.session_state["deleted_account_summary"] = summary
                st.session_state["auth_user"] = None
                st.rerun()
            except ValueError as err:
                st.error(str(err))

    st.divider()
    if st.button("🚪 Sign Out", use_container_width=True):
        clear_user_session_cache()
        st.session_state.pop("new_recovery_kit", None)
        st.session_state["auth_user"] = None
        st.rerun()


# ============================================================================
# Top Workspace Header & Status Banners
# ============================================================================
st.markdown("## 🛡️ LocalFinAI Workspace")

if st.session_state.pop("pw_reset_success_banner", False):
    st.success("Your password has been reset and you are now signed in to your workspace.")

restore_stats = st.session_state.pop("restore_success_banner", None)
if restore_stats:
    st.success(
        f"Workspace Backup Restored! Imported **{restore_stats['restored_transactions']}** transactions "
        f"(skipped {restore_stats['skipped_duplicates']} duplicates), "
        f"**{restore_stats['restored_rules']}** merchant category rules, "
        f"**{restore_stats['restored_forecasts']}** monthly forecasts, and "
        f"RL PPO Model: **{'Yes' if restore_stats['restored_rl_model'] else 'No'}**."
    )

if "new_recovery_kit" in st.session_state:
    kit = st.session_state["new_recovery_kit"]
    st.warning(
        f"🔐 **Save Your Master Recovery Key:** `{kit['recovery_key']}`\n\n"
        "Because LocalFinAI runs 100% offline with zero external servers, this key (or your Security Question) "
        "is the only way to recover your account if you forget your password."
    )
    kit_col1, kit_col2, _ = st.columns([1.6, 1.2, 2.2])
    with kit_col1:
        kit_txt = build_recovery_kit_text(
            username=kit["username"],
            display_name=kit["display_name"],
            recovery_key=kit["recovery_key"],
            question=kit["security_question"],
        )
        st.download_button(
            "⬇️ Download Emergency Recovery Kit (.txt)",
            data=kit_txt.encode("utf-8"),
            file_name=f"LocalFinAI_Recovery_Kit_{kit['username']}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True,
        )
    with kit_col2:
        if st.button("✓ I Have Saved My Recovery Key", use_container_width=True):
            st.session_state.pop("new_recovery_kit", None)
            st.rerun()

tab_ledger, tab_summary, tab_rl, tab_mcp = st.tabs(
    [
        "📒 Ledger & Statement Import",
        "📊 Spend Analytics & AI Insights",
        "📈 RL Expense Forecaster",
        "🧭 MCP Investment Advisor",
    ]
)

# ----------------------------------------------------------------------------
# TAB 1: Statement Ingestion & Full CRUD Ledger
# ----------------------------------------------------------------------------
with tab_ledger:
    ledger_df = get_transactions_dataframe(user_id=USER_ID, verified_only=True)

    has_staging = "staging_df" in st.session_state and not st.session_state["staging_df"].empty
    with st.expander(
        "📥 Import Bank Statement (PDF / CSV / Image) or Stage Batch Entries",
        expanded=(ledger_df.empty or has_staging),
    ):
        up_col1, up_col2 = st.columns([2.2, 1.2])
        with up_col1:
            uploaded_file = st.file_uploader(
                "Select Statement File (Native PDF, Scanned PDF, CSV, PNG, JPG)",
                type=["pdf", "csv", "png", "jpg", "jpeg"],
            )
        with up_col2:
            pdf_password = ""
            if uploaded_file and uploaded_file.name.lower().endswith(".pdf"):
                pdf_password = st.text_input(
                    "PDF Password (if encrypted)",
                    type="password",
                    help="Enter your bank PDF password (e.g., Customer ID, DOB, or PAN).",
                )
            st.write("")
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                run_extraction = st.button(
                    "Extract Statement",
                    type="primary",
                    disabled=uploaded_file is None,
                    use_container_width=True,
                )
            with btn_c2:
                if st.button("New Staging Sheet", use_container_width=True):
                    st.session_state["staging_df"] = pd.DataFrame(
                        [
                            {
                                "delete_row": False,
                                "date": date.today(),
                                "merchant": "",
                                "raw_description": "",
                                "amount": 0.0,
                                "tx_type": "debit",
                                "category": "Miscellaneous",
                                "is_recurring": False,
                                "source": "manual",
                                "categorization_tier": "Manual",
                            }
                        ]
                    )
                    st.session_state["initial_ai_categories"] = {}
                    st.rerun()

        if run_extraction and uploaded_file is not None:
            extractor = StatementExtractor()
            categorizer = HybridCategorizer(user_id=USER_ID)
            file_bytes = uploaded_file.read()
            fname = uploaded_file.name.lower()

            try:
                with st.spinner("Extracting lines & running 3-Tier local categorizer..."):
                    if fname.endswith(".csv"):
                        extracted_rows = extractor.parse_csv_bytes(file_bytes)
                    elif fname.endswith(".pdf"):
                        raw_lines = extractor.extract_raw_lines_from_pdf(
                            file_bytes, password=pdf_password
                        )
                        extracted_rows = extractor.parse_lines_with_hf_llm(raw_lines)
                    else:
                        raw_lines = extractor.extract_raw_lines_from_image(file_bytes)
                        extracted_rows = extractor.parse_lines_with_hf_llm(raw_lines)

                    categorized_rows = categorizer.categorize_batch(extracted_rows)

                if categorized_rows:
                    stg_df = pd.DataFrame(categorized_rows)
                    stg_df["date"] = pd.to_datetime(stg_df["date"]).dt.date
                    stg_df.insert(0, "delete_row", False)
                    st.session_state["staging_df"] = stg_df
                    st.session_state["initial_ai_categories"] = {
                        idx: row["category"] for idx, row in enumerate(categorized_rows)
                    }
                    st.success(
                        f"Extracted {len(stg_df)} candidate rows. Review and edit below before committing."
                    )
                else:
                    st.warning("No valid transaction lines were detected in this file.")
            except ValueError as ve:
                st.error(str(ve))
            except Exception as exc:
                st.error(f"Extraction error: {exc}")

        if "staging_df" in st.session_state and not st.session_state["staging_df"].empty:
            st.markdown("##### Staging Table (Review, Edit, or Remove Rows Before Committing)")
            edited_staging_df = st.data_editor(
                st.session_state["staging_df"],
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "delete_row": st.column_config.CheckboxColumn("🗑️ Skip?", default=False),
                    "date": st.column_config.DateColumn("Date", required=True, default=date.today()),
                    "merchant": st.column_config.TextColumn("Merchant", required=True, default=""),
                    "raw_description": st.column_config.TextColumn("Description", default=""),
                    "amount": st.column_config.NumberColumn(
                        f"Amount ({CURRENCY})",
                        min_value=0.0,
                        format="%.2f",
                        required=True,
                        default=0.0,
                    ),
                    "tx_type": st.column_config.SelectboxColumn(
                        "Type", options=["debit", "credit"], required=True, default="debit"
                    ),
                    "category": st.column_config.SelectboxColumn(
                        "Category",
                        options=ALL_CATEGORIES,
                        required=True,
                        default="Miscellaneous",
                    ),
                    "is_recurring": st.column_config.CheckboxColumn("Recurring?", default=False),
                    "source": st.column_config.SelectboxColumn(
                        "Source", options=["ai_extracted", "manual"], default="manual"
                    ),
                    "categorization_tier": st.column_config.TextColumn("Tier", disabled=True),
                },
                key="staging_editor_widget",
            )

            sc1, sc2, _ = st.columns([1.6, 1.0, 2.4])
            with sc1:
                if st.button("✅ Commit Rows to Ledger", type="primary", use_container_width=True):
                    records_to_commit = edited_staging_df.to_dict(orient="records")
                    result = commit_verified_transactions(
                        user_id=USER_ID,
                        staged_rows=records_to_commit,
                        initial_ai_categories=st.session_state.get("initial_ai_categories", {}),
                    )
                    clear_user_session_cache()
                    st.success(
                        f"Saved {result['inserted']} transactions and learned {result['learned_corrections']} category rules."
                    )
                    st.rerun()
            with sc2:
                if st.button("Discard Staging", use_container_width=True):
                    st.session_state.pop("staging_df", None)
                    st.rerun()

    # ------------------------------------------------------------------------
    # Full CRUD Verified SQLite Ledger
    # ------------------------------------------------------------------------
    st.markdown("#### Verified Transactions Ledger")

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.8, 1.4, 1.0, 1.0])
    with f_col1:
        search_query = st.text_input(
            "Search Merchant or Description",
            placeholder="Filter by merchant or keyword...",
            label_visibility="collapsed",
        )
    with f_col2:
        category_filter = st.multiselect(
            "Filter Categories",
            options=ALL_CATEGORIES,
            placeholder="All Categories",
            label_visibility="collapsed",
        )
    with f_col3:
        type_filter = st.selectbox(
            "Filter Type",
            options=["All Types", "debit", "credit"],
            label_visibility="collapsed",
        )
    with f_col4:
        if not ledger_df.empty:
            csv_bytes = (
                ledger_df[
                    ["date", "merchant", "raw_description", "amount", "tx_type", "category", "is_recurring", "source"]
                ]
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                "⬇️ Export CSV",
                data=csv_bytes,
                file_name=f"localfinai_ledger_{current_user['username']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    filtered_df = ledger_df.copy()
    if not filtered_df.empty:
        if search_query.strip():
            q = search_query.strip().lower()
            mask = (
                filtered_df["merchant"].astype(str).str.lower().str.contains(q, na=False)
                | filtered_df["raw_description"].astype(str).str.lower().str.contains(q, na=False)
            )
            filtered_df = filtered_df[mask]
        if category_filter:
            filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]
        if type_filter != "All Types":
            filtered_df = filtered_df[filtered_df["tx_type"] == type_filter]

    if filtered_df.empty:
        st.caption(
            "Your ledger is currently blank (or no rows match your filter). "
            "Click the `+` row below to add transactions directly, or upload a statement above."
        )
        editable_ledger = pd.DataFrame(
            columns=[
                "delete_row",
                "id",
                "date",
                "merchant",
                "raw_description",
                "amount",
                "tx_type",
                "category",
                "is_recurring",
                "source",
            ]
        )
        visible_ids = []
    else:
        visible_ids = [int(i) for i in filtered_df["id"].dropna().tolist()]
        editable_ledger = filtered_df[
            [
                "id",
                "date",
                "merchant",
                "raw_description",
                "amount",
                "tx_type",
                "category",
                "is_recurring",
                "source",
            ]
        ].copy()
        editable_ledger["date"] = pd.to_datetime(editable_ledger["date"]).dt.date
        editable_ledger.insert(0, "delete_row", False)

    edited_ledger_df = st.data_editor(
        editable_ledger,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "delete_row": st.column_config.CheckboxColumn(
                "🗑️ Delete?",
                help="Check this box and click 'Save Ledger Changes' to delete the row.",
                default=False,
            ),
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "date": st.column_config.DateColumn("Date", required=True, default=date.today()),
            "merchant": st.column_config.TextColumn("Merchant", required=True, default=""),
            "raw_description": st.column_config.TextColumn("Description", default=""),
            "amount": st.column_config.NumberColumn(
                f"Amount ({CURRENCY})",
                min_value=0.01,
                format="%.2f",
                required=True,
                default=100.0,
            ),
            "tx_type": st.column_config.SelectboxColumn(
                "Type", options=["debit", "credit"], required=True, default="debit"
            ),
            "category": st.column_config.SelectboxColumn(
                "Category",
                options=ALL_CATEGORIES,
                required=True,
                default="Miscellaneous",
            ),
            "is_recurring": st.column_config.CheckboxColumn("Recurring?", default=False),
            "source": st.column_config.SelectboxColumn(
                "Source", options=["ai_extracted", "manual"], default="manual"
            ),
        },
        key="main_crud_ledger_editor",
    )

    act_col1, act_col2, _ = st.columns([1.8, 1.1, 2.1])
    with act_col1:
        if st.button(
            "💾 Save Ledger Changes (Add / Edit / Delete)",
            type="primary",
            use_container_width=True,
        ):
            crud_stats = sync_ledger_crud(
                user_id=USER_ID,
                edited_df=edited_ledger_df,
                visible_ids=visible_ids,
            )
            clear_user_session_cache()
            st.success(
                f"Saved! Added: {crud_stats['inserted']} • Updated: {crud_stats['updated']} • "
                f"Deleted: {crud_stats['deleted']} • Rules Learned: {crud_stats['learned']}"
            )
            st.rerun()
    with act_col2:
        if not ledger_df.empty:
            if st.button("🧹 Clear All Rows", use_container_width=True):
                removed = clear_all_transactions(USER_ID)
                clear_user_session_cache()
                st.warning(f"Deleted all {removed} transactions from your workspace.")
                st.rerun()


# ----------------------------------------------------------------------------
# TAB 2: Spend Analytics & AI Insights
# ----------------------------------------------------------------------------
with tab_summary:
    analytics = compute_spend_analytics(user_id=USER_ID, currency_symbol=CURRENCY)

    if not analytics["has_data"]:
        st.info(
            "Your analytics dashboard is currently blank. Add or upload transactions in the **Ledger & Statement Import** tab to view cashflow metrics, category shares, and AI commentary."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            f"Income ({analytics['latest_month']})",
            f"{CURRENCY}{analytics['total_income']:,.2f}",
        )
        m2.metric(
            f"Expenses ({analytics['latest_month']})",
            f"{CURRENCY}{analytics['total_spend']:,.2f}",
        )
        m3.metric("Net Cashflow", f"{CURRENCY}{analytics['net_cashflow']:,.2f}")
        m4.metric("Savings Rate", f"{analytics['savings_rate_pct']:.1f}%")

        st.write("")
        cat_table: pd.DataFrame = analytics["category_table"]
        if not cat_table.empty:
            chart_col, table_col = st.columns([1.4, 2.0])
            with chart_col:
                st.markdown("##### Spend by Category")
                chart_df = cat_table.set_index("category")[["current_spend"]]
                st.bar_chart(chart_df, height=300)
            with table_col:
                st.markdown(f"##### Category Breakdown & Month-over-Month Deltas ({CURRENCY})")
                st.dataframe(cat_table, use_container_width=True, hide_index=True)

        if not analytics["top_merchants"].empty:
            st.markdown(f"##### Top Merchants ({analytics['latest_month']})")
            st.dataframe(analytics["top_merchants"], use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("##### Local Qwen2.5-3B Executive Commentary")
        if st.button("✨ Generate AI Spend Analysis", type="primary"):
            with st.spinner("Analyzing deterministic SQL metrics with local Qwen2.5-3B..."):
                st.session_state["ai_commentary_text"] = generate_ai_spend_commentary(
                    analytics, currency_symbol=CURRENCY
                )

        if st.session_state.get("ai_commentary_text"):
            st.info(st.session_state["ai_commentary_text"])


# ----------------------------------------------------------------------------
# TAB 3: RL Expense Forecaster
# ----------------------------------------------------------------------------
with tab_rl:
    rl_manager = RLForecastManager(user_id=USER_ID)
    user_tx_df = get_transactions_dataframe(user_id=USER_ID, verified_only=True)
    user_debits = user_tx_df[user_tx_df["tx_type"] == "debit"] if not user_tx_df.empty else pd.DataFrame()

    if user_debits.empty:
        st.info(
            "Your RL Expense Forecaster is currently blank. Add debit transactions in the **Ledger & Statement Import** tab first so the PPO agent can compute your 3-month rolling baselines and next-month category forecasts."
        )
    else:
        col_r1, col_r2, _ = st.columns([1.5, 1.5, 1.5])
        with col_r1:
            if st.button(
                "🔮 Predict Next-Month Expenses",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Running PPO policy inference on your ledger categories..."):
                    st.session_state["latest_rl_pred_df"] = rl_manager.predict_next_month_expenses()
        with col_r2:
            if st.button(
                "🧠 Train / Fine-Tune PPO Policy",
                use_container_width=True,
                help="Unloads the GGUF LLM to free RAM, then fine-tunes your personal PPO agent on CPU.",
            ):
                with st.spinner("Unloading LLM & fine-tuning personal PPO agent on CPU..."):
                    metrics = rl_manager.train_or_finetune(
                        total_timesteps=4096, use_synthetic_bootstrap=True
                    )
                    st.session_state["latest_rl_pred_df"] = rl_manager.predict_next_month_expenses()
                st.success(
                    f"{metrics['mode']} complete • Mean Reward: {metrics['mean_reward']:.4f}"
                )

        pred_df = st.session_state.get("latest_rl_pred_df", pd.DataFrame())
        if not pred_df.empty:
            total_baseline = float(pred_df["baseline_spend"].sum())
            total_rl_pred = float(pred_df["predicted_spend"].sum())

            k1, k2, k3 = st.columns(3)
            k1.metric("Forecast Target Month", str(pred_df["year_month"].iloc[0]))
            k2.metric("Statistical Baseline (3M WMA)", f"{CURRENCY}{total_baseline:,.2f}")
            k3.metric(
                "RL-Adjusted Forecast",
                f"{CURRENCY}{total_rl_pred:,.2f}",
                f"{((total_rl_pred - total_baseline) / max(total_baseline, 1.0)) * 100:+.1f}% vs Baseline",
            )
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        else:
            st.caption(
                "Click **🔮 Predict Next-Month Expenses** above to generate your category-by-category forecast."
            )

        hist_forecasts = get_forecasts_dataframe(user_id=USER_ID)
        if not hist_forecasts.empty:
            with st.expander("📜 Historical Forecast vs. Actuals Log", expanded=False):
                st.dataframe(hist_forecasts, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# TAB 4: Local MCP Investment Advisor
# ----------------------------------------------------------------------------
with tab_mcp:
    st.caption(
        "Personal transactions never leave your machine. Only your aggregate monthly surplus and risk profile are passed to the isolated MCP server (`yfinance` + RSS news + deterministic SIP allocator)."
    )

    analytics = compute_spend_analytics(user_id=USER_ID, currency_symbol=CURRENCY)
    pred_df = st.session_state.get("latest_rl_pred_df", pd.DataFrame())

    default_income = float(analytics["total_income"]) if analytics["has_data"] else 0.0
    default_pred_exp = (
        float(pred_df["predicted_spend"].sum())
        if not pred_df.empty
        else (float(analytics["total_spend"]) if analytics["has_data"] else 0.0)
    )

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        user_income = st.number_input(
            f"Monthly Net Income ({CURRENCY})",
            min_value=0.0,
            value=default_income,
            step=1000.0 if CURRENCY == "₹" else 100.0,
        )
    with c_m2:
        rl_pred_exp = st.number_input(
            f"Projected Next-Month Expenses ({CURRENCY})",
            min_value=0.0,
            value=default_pred_exp,
            step=500.0 if CURRENCY == "₹" else 50.0,
        )
    with c_m3:
        risk_profile = st.selectbox(
            "Risk Tolerance Profile",
            ["Moderate", "Conservative", "Aggressive"],
        )

    investable_surplus = max(0.0, user_income - rl_pred_exp)
    st.metric("Projected Monthly Investable Surplus", f"{CURRENCY}{investable_surplus:,.2f}")

    user_goal = st.text_input(
        "Investment Goal & Preferences (Optional)",
        value="",
        placeholder=(
            "e.g., Build a 6-month emergency fund and invest monthly surplus into Nifty 50, Gold, and Liquid ETFs."
            if CURRENCY == "₹"
            else "e.g., Build a 6-month emergency buffer and invest surplus into low-cost index and Treasury ETFs."
        ),
    )

    if st.button(
        "🧭 Run MCP Tool Router & Generate Investment Plan",
        type="primary",
        disabled=(user_income <= 0 and rl_pred_exp <= 0),
    ):
        orchestrator = MCPInvestmentOrchestrator()
        effective_goal = user_goal.strip() or (
            "Build a 6-month emergency fund and invest monthly surplus via SIP into diversified index, debt, and gold ETFs."
        )
        with st.spinner("Querying local MCP tools (yfinance + RSS feeds + SIP calculator) & synthesizing plan..."):
            st.session_state["mcp_advisor_result"] = orchestrator.run_advisor(
                monthly_income=user_income,
                predicted_expenses=rl_pred_exp,
                risk_profile=risk_profile,
                user_goal=effective_goal,
                currency_symbol=CURRENCY,
            )

    if user_income <= 0 and rl_pred_exp <= 0 and "mcp_advisor_result" not in st.session_state:
        st.info(
            "Enter your **Monthly Net Income** and **Projected Next-Month Expenses** above (or add transactions in the Ledger tab so they populate automatically) to run the MCP Investment Advisor."
        )

    if "mcp_advisor_result" in st.session_state:
        mcp_result = st.session_state["mcp_advisor_result"]
        st.markdown("#### Personalized Fiduciary Recommendation")
        st.success(mcp_result["advisor_recommendation"])

        with st.expander("🔍 Inspect MCP Tool Calls, SIP Allocation & Live Market Data", expanded=False):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("##### LLM Structured `ToolCallPlan`")
                st.json(mcp_result["tool_call_plan"])
                st.markdown("##### `calculate_investment_allocation` Output")
                st.json(mcp_result["allocation_output"])
            with col_t2:
                st.markdown("##### `get_asset_metrics` (`yfinance`)")
                st.json(mcp_result["asset_metrics_output"])
                st.markdown("##### `fetch_market_news` (`feedparser` RSS)")
                st.json(mcp_result["market_news_output"])