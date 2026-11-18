"""Transactions page: upload, validate, filter, and correct categories."""
import pandas as pd
import streamlit as st

from utils import financial_analyzer as fa
from utils import ui
from utils.categorizer import ALL_CATEGORIES
from utils.data_processor import SAMPLE_PATH, ValidationError, clean_transactions


def _handle_upload() -> None:
    up = st.file_uploader("Upload your bank statement (CSV)", type=["csv"])
    if up is not None:
        file_id = f"{up.name}-{up.size}"
        if st.session_state.get("loaded_file_id") != file_id:
            try:
                raw = pd.read_csv(up)
                df, notes = clean_transactions(raw)
            except ValidationError as e:
                st.error(str(e))
                return
            except pd.errors.EmptyDataError:
                st.error("The file is empty or is not a valid CSV.")
                return
            except Exception as e:  # unreadable encoding, malformed CSV, etc.
                st.error(f"Could not read this file: {e}")
                return
            st.session_state.df = df
            st.session_state.data_source = up.name
            st.session_state.loaded_file_id = file_id
            st.session_state.upload_notes = notes
            st.rerun()
        st.success(f"Loaded **{up.name}**: {len(st.session_state.df):,} transactions.")
    for note in st.session_state.get("upload_notes", []):
        st.warning(note)


def render() -> None:
    ui.page_header("Transactions", "Upload a statement, review transactions and fix categories")
    cur = st.session_state.get("currency", "₹")

    with ui.card("upload"):
        ui.section_title("Bank statement upload")
        st.caption("Your file stays on your computer. It is processed locally and never uploaded anywhere.")
        _handle_upload()
        with st.expander("Expected CSV format"):
            st.markdown(
                "Required columns: **date**, **description**, **amount**. "
                "Add a **type** column (`Income` / `Expense`), or use negative amounts for expenses. "
                "Separate *debit* / *credit* columns are also supported. "
                "An optional **category** column is respected."
            )
            if SAMPLE_PATH.exists():
                st.download_button("Download sample CSV", SAMPLE_PATH.read_bytes(),
                                   "sample_transactions.csv", "text/csv")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info("Upload a CSV above to get started.")
        return

    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions", f"{len(df):,}")
    k2.metric("Income entries", f"{int((df['type'] == 'Income').sum()):,}")
    k3.metric("Expense entries", f"{int((df['type'] == 'Expense').sum()):,}")
    k4.metric("Categorised 'Other'", f"{int(((df['category'] == 'Other') & (df['type'] == 'Expense')).sum()):,}")

    st.write("")
    with ui.card("table"):
        ui.section_title("All transactions")
        months = sorted(df["month"].unique())
        labels = {fa.pretty_month(m): m for m in months}
        f1, f2, f3, f4 = st.columns([1, 1, 2, 2])
        month = f1.selectbox("Month", ["All"] + list(reversed(list(labels))))
        ttype = f2.selectbox("Type", ["All", "Income", "Expense"])
        cats = f3.multiselect("Category", ALL_CATEGORIES)
        query = f4.text_input("Search description", placeholder="e.g. swiggy")

        view = df.copy()
        if month != "All":
            view = view[view["month"] == labels[month]]
        if ttype != "All":
            view = view[view["type"] == ttype]
        if cats:
            view = view[view["category"].isin(cats)]
        if query:
            view = view[view["description"].str.contains(query, case=False, na=False)]
        view = view.sort_values("date", ascending=False)
        shown = view[["date", "description", "amount", "type", "category"]]

        st.caption(f"{len(shown):,} transactions shown. Edit the **Category** column, then save.")
        sig = f"{month}|{ttype}|{'-'.join(cats)}|{query}|{len(df)}"
        edited = st.data_editor(
            shown,
            key=f"txn_editor_{abs(hash(sig))}",
            hide_index=True,
            num_rows="fixed",
            disabled=["date", "description", "amount", "type"],
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "description": "Description",
                "amount": st.column_config.NumberColumn("Amount", format=f"{cur}%.2f"),
                "type": "Type",
                "category": st.column_config.SelectboxColumn("Category", options=ALL_CATEGORIES, required=True),
            },
            **ui.stretch(),
        )

        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("Save category changes", type="primary"):
            changed = edited["category"] != shown["category"]
            n = int(changed.sum())
            if n:
                st.session_state.df.loc[edited.index[changed], "category"] = edited.loc[changed, "category"]
                st.toast(f"Updated {n} categor{'y' if n == 1 else 'ies'}.", icon="✅")
                st.rerun()
            else:
                st.toast("No changes to save.")
        b2.download_button("Download CSV", view.drop(columns=["month"]).to_csv(index=False).encode(),
                           "finwise_transactions.csv", "text/csv")
