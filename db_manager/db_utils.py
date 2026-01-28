"""
db_utils.py

Simple sqlite3 helper utilities:
- create_table(db_name, table_name, columns)
- insert_row(db_name, table_name, values)
- read_table(db_name, table_name)
- get_a_row(db_name, table_name, primary_col, value)
- remove_row(db_name, table_name, primary_col, value)
- sanitize_column_name(list_of_column_name)
"""

from typing import List, Tuple, Dict, Any, Optional
import sqlite3
import re
import os


# ---------------------
# Utility helpers
# ---------------------
def _db_path(db_name: str) -> str:
    """Return database filename with .db appended if necessary."""
    if db_name.endswith(".db"):
        return db_name
    return f"{db_name}.db"


def sanitize_column_name(cols: List[str]) -> List[str]:
    """
    Replace special characters (including spaces) with underscores and collapse repeated underscores.
    Ensure resulting names are non-empty. Preserve case.
    For example: "Gene Symbol" -> "Gene_Symbol", "a!b" -> "a_b"
    """
    sanitized = []
    for c in cols:
        if c is None:
            name = ""
        else:
            name = str(c).strip()
        # replace non-alphanumeric and non-underscore with "_"
        name = re.sub(r"[^0-9A-Za-z_]", "_", name)
        # collapse multiple underscores
        name = re.sub(r"_+", "_", name)
        # strip leading/trailing underscores
        name = name.strip("_")
        # if empty after sanitization, give a default name
        if name == "":
            name = "col"
        sanitized.append(name)
    # make unique if collisions occurred
    seen = {}
    unique = []
    for s in sanitized:
        base = s
        if base not in seen:
            seen[base] = 0
            unique.append(base)
        else:
            seen[base] += 1
            new_name = f"{base}_{seen[base]}"
            # ensure new_name is unique in turn
            while new_name in seen:
                seen[base] += 1
                new_name = f"{base}_{seen[base]}"
            seen[new_name] = 0
            unique.append(new_name)
    return unique


def _quote_identifier(name: str) -> str:
    """
    Safely quote an identifier for SQLite - we expect name already sanitized to [A-Za-z0-9_]+
    We'll wrap it in double quotes for safety.
    """
    return f'"{name}"'


def _open_conn(db_name: str) -> sqlite3.Connection:
    path = _db_path(db_name)
    # ensure folder exists
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[Tuple[str, str, int]]:
    """
    Return list of (name, type, pk) for columns in table (in declared order).
    pk is 1 if part of primary key, else 0.
    """
    cur = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})")
    rows = cur.fetchall()
    return [(r[1], r[2], r[5]) for r in rows]  # (name, type, pk)


# ---------------------
# Public functions
# ---------------------
def create_table(db_name: str, table_name: str, columns: List[Tuple[str, str]]) -> None:
    """
    Create a table in the database.

    Parameters:
      - db_name: database name WITHOUT .db ('.db' will be appended automatically)
      - table_name: table name (string)
      - columns: list of tuples (column_display_name, data_type) in order.
                 The first column becomes the PRIMARY KEY.
                 Example: [("ID", "TEXT"), ("Gene", "TEXT"), ("Value", "REAL")]

    Raises:
      - ValueError on invalid input
    """
    if not columns or len(columns) == 0:
        raise ValueError("columns must be a non-empty list of (name, type) tuples")

    # sanitize column names and ensure uniqueness
    orig_names = [c[0] for c in columns]
    types = [c[1] for c in columns]
    sanitized = sanitize_column_name(orig_names)

    # Build CREATE TABLE SQL
    col_defs = []
    for i, (col_name, col_type) in enumerate(zip(sanitized, types)):
        if i == 0:
            # first column -> primary key
            col_defs.append(f"{_quote_identifier(col_name)} {col_type} PRIMARY KEY")
        else:
            col_defs.append(f"{_quote_identifier(col_name)} {col_type}")

    sql = f"CREATE TABLE IF NOT EXISTS {_quote_identifier(table_name)} ({', '.join(col_defs)});"

    conn = _open_conn(db_name)
    try:
        conn.execute(sql)
        conn.commit()
    finally:
        conn.close()

def update_table(db_name: str, table_name: str, values: Dict[str, list]) -> int:
    """
    Upsert rows in table using primary key.

    For each index i in the provided lists:
      - if a row with primary-key == values[pk][i] exists, UPDATE it (only columns provided)
      - otherwise INSERT a new row with the provided columns

    Returns:
      Number of rows inserted or updated (total).
    """
    if not isinstance(values, dict) or not values:
        raise ValueError("values must be a non-empty dict of lists")

    # Ensure all values are lists of equal length
    lengths = {len(v) for v in values.values()}
    if len(lengths) != 1:
        raise ValueError("All value lists must have the same length")

    num_rows = lengths.pop()

    conn = _open_conn(db_name)
    try:
        # Get table schema
        cols_info = _get_table_columns(conn, table_name)
        if not cols_info:
            raise ValueError(f"Table '{table_name}' does not exist")

        table_cols = [c[0] for c in cols_info]
        pk_cols = [c[0] for c in cols_info if c[2] == 1]
        pk_col = pk_cols[0] if pk_cols else table_cols[0]

        # Map user-supplied keys (original or unsanitized) -> sanitized table column names
        input_keys = list(values.keys())
        sanitized_input_keys = sanitize_column_name(input_keys)
        # input_map: sanitized_key -> original user key
        input_map = {s: k for s, k in zip(sanitized_input_keys, input_keys)}

        # Ensure primary key is provided by the caller (after sanitization matching)
        if pk_col not in input_map:
            raise ValueError(f"Primary key column '{pk_col}' must be included in values")

        total_changed = 0

        for i in range(num_rows):
            # Determine primary key value for this row
            pk_value = values[input_map[pk_col]][i]

            # Check existence
            cur = conn.execute(
                f"SELECT 1 FROM {_quote_identifier(table_name)} WHERE {_quote_identifier(pk_col)} = ? LIMIT 1;",
                (pk_value,)
            )
            exists = cur.fetchone() is not None

            if exists:
                # Build UPDATE: only for table columns present in input_map and not the pk
                set_clauses = []
                params = []
                for tbl_col in table_cols:
                    if tbl_col == pk_col:
                        continue
                    if tbl_col in input_map:
                        orig_key = input_map[tbl_col]
                        set_clauses.append(f"{_quote_identifier(tbl_col)} = ?")
                        params.append(values[orig_key][i])

                if not set_clauses:
                    # nothing provided to update for this row
                    continue

                sql = f"""
                    UPDATE {_quote_identifier(table_name)}
                    SET {', '.join(set_clauses)}
                    WHERE {_quote_identifier(pk_col)} = ?;
                """
                cur = conn.execute(sql, (*params, pk_value))
                # sqlite3's rowcount reports number of rows modified by the statement
                total_changed += cur.rowcount if cur.rowcount is not None else 0

            else:
                # Build INSERT: include all table columns that the caller provided
                insert_cols = []
                insert_vals = []
                for tbl_col in table_cols:
                    if tbl_col in input_map:
                        orig_key = input_map[tbl_col]
                        insert_cols.append(_quote_identifier(tbl_col))
                        insert_vals.append(values[orig_key][i])

                # Primary key must be in insert_cols (we already validated that earlier)
                if not insert_cols:
                    # No columns to insert (shouldn't happen because pk must be present)
                    continue

                placeholders = ", ".join(["?"] * len(insert_vals))
                cols_sql = ", ".join(insert_cols)
                sql = f"INSERT INTO {_quote_identifier(table_name)} ({cols_sql}) VALUES ({placeholders});"
                conn.execute(sql, tuple(insert_vals))
                total_changed += 1

        conn.commit()
        return total_changed

    finally:
        conn.close()


def insert_row(db_name: str, table_name: str, values: Dict[str, Any]) -> bool:
    """
    Insert a row into table. Checks for duplicate primary column value and returns False if duplicate.
    Returns True when row inserted successfully.

    values: dict mapping original column names (or sanitized names) -> value
            e.g. {"ID": "abc", "Gene": "TP53", "Text": "some text"}
    """
    if not isinstance(values, dict):
        raise ValueError("values must be a dict mapping column->value")

    conn = _open_conn(db_name)
    try:
        # get columns info
        cols_info = _get_table_columns(conn, table_name)
        if not cols_info:
            raise ValueError(f"Table '{table_name}' does not exist or has no columns.")

        table_cols = [c[0] for c in cols_info]
        pk_cols = [c[0] for c in cols_info if c[2] == 1]
        if len(pk_cols) == 0:
            # if no PK flagged (unlikely with create_table), consider first column as pk
            pk_col = table_cols[0]
        else:
            pk_col = pk_cols[0]

        # Build mapping from lower(original) sanitized forms to actual table column names
        # Allow user to pass either original name or sanitized name by normalizing keys
        # We'll sanitize the provided keys as well to attempt matching.
        input_keys = list(values.keys())
        sanitized_input_keys = sanitize_column_name(input_keys)
        input_map = {s: k for s, k in zip(sanitized_input_keys, input_keys)}
        # Now map to table columns
        # Only include keys that match a table column
        final_cols = []
        final_vals = []
        for tbl_col in table_cols:
            # try to find corresponding input key (by sanitized name)
            if tbl_col in input_map:
                orig_key = input_map[tbl_col]
                final_cols.append(tbl_col)
                final_vals.append(values[orig_key])
            else:
                # also try matching ignoring case
                matched = None
                for orig_key in input_keys:
                    if sanitize_column_name([orig_key])[0].lower() == tbl_col.lower():
                        matched = orig_key
                        break
                if matched is not None:
                    final_cols.append(tbl_col)
                    final_vals.append(values[matched])
                # else: missing column -> will be omitted (use default/null)

        if not final_cols:
            raise ValueError("No matching columns found between provided values and table schema.")

        # check duplicate primary key
        # we require the primary column value to be present in the provided dict
        if pk_col not in final_cols:
            raise ValueError(f"Primary key column '{pk_col}' must be present in values")
        pk_index = final_cols.index(pk_col)
        pk_value = final_vals[pk_index]
        # Check existing
        cur = conn.execute(
            f"SELECT 1 FROM {_quote_identifier(table_name)} WHERE {_quote_identifier(pk_col)} = ? LIMIT 1;",
            (pk_value,)
        )
        if cur.fetchone() is not None:
            # duplicate
            mgs = f"Duplicate primary key value '{pk_value}' found in table '{table_name}'"
            return False, mgs

        # Build insert
        cols_sql = ", ".join([_quote_identifier(c) for c in final_cols])
        placeholders = ", ".join(["?"] * len(final_cols))
        sql = f"INSERT INTO {_quote_identifier(table_name)} ({cols_sql}) VALUES ({placeholders});"
        conn.execute(sql, tuple(final_vals))
        conn.commit()
        mgs = f"Row inserted into table '{table_name}'"
        return True, mgs
    finally:
        conn.close()


def read_table(db_name: str, table_name: str) -> Dict[str, List[Any]]:
    """
    Read an entire table and return  a dict-of-lists: ["col 1": [col 1 values...], "col 2": [col 2 values...], ...]
    """
    conn = _open_conn(db_name)
    try:
        cur = conn.execute(f"SELECT * FROM {_quote_identifier(table_name)};")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        # transpose rows into column lists
        result = {col: [] for col in cols}
        for r in rows:
            for idx, col in enumerate(cols):
                result[col].append(r[idx])
        return result
    finally:
        conn.close()


def get_a_row(db_name: str, table_name: str, primary_col: str, value: Any) -> Optional[Dict[str, Any]]:
    """
    Return one row where primary_col == value as a dict {col: value, ...}. Returns None if not found.
    primary_col can be either original or sanitized; sanitized internally.
    """
    conn = _open_conn(db_name)
    try:
        # sanitize column name
        sanitized_primary = sanitize_column_name([primary_col])[0]
        cur = conn.execute(
            f"SELECT * FROM {_quote_identifier(table_name)} WHERE {_quote_identifier(sanitized_primary)} = ? LIMIT 1;",
            (value,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return {col: row[idx] for idx, col in enumerate(cols)}
    finally:
        conn.close()


def remove_row(db_name: str, table_name: str, primary_col: str, value: Any) -> bool:
    """
    Remove a row where primary_col == value. Returns True if a row was deleted, False otherwise.
    """
    conn = _open_conn(db_name)
    try:
        sanitized_primary = sanitize_column_name([primary_col])[0]
        cur = conn.execute(
            f"DELETE FROM {_quote_identifier(table_name)} WHERE {_quote_identifier(sanitized_primary)} = ?;",
            (value,)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()

def table_exists(db_name: str, table_name: str) -> bool:
    """
    Return True if `table_name` exists in the database `db_name`, False otherwise.

    Parameters:
      - db_name: database name WITHOUT .db ('.db' will be appended automatically)
      - table_name: name of the table to check

    Notes:
      - Uses sqlite_master lookup first for an exact match, and falls back to a
        sanitized-name check if the original name doesn't match (to handle cases
        where callers might pass sanitized vs original names).
    """
    if not table_name or not isinstance(table_name, str):
        raise ValueError("table_name must be a non-empty string")

    conn = _open_conn(db_name)
    try:
        # First try an exact match against sqlite_master (works for normal table names,
        # including those with spaces or special characters).
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1;",
            (table_name,)
        )
        if cur.fetchone() is not None:
            return True

        # Fallback: try sanitized variant (in case the table was created using a sanitized name)
        sanitized_name = sanitize_column_name([table_name])[0]
        if sanitized_name != table_name:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1;",
                (sanitized_name,)
            )
            if cur.fetchone() is not None:
                return True

        return False
    finally:
        conn.close()

# ---------------------
# If run as script, quick demo
# ---------------------
if __name__ == "__main__":
    # Quick demonstration of usage
    demo_db = "demo_db"
    demo_table = "variants"
    columns = [("ID", "TEXT"), ("Gene Symbol", "TEXT"), ("Variation", "TEXT"), ("Text", "TEXT")]

    print("Creating table...")
    create_table(demo_db, demo_table, columns)

    print("Inserting row 1...")
    ok = insert_row(demo_db, demo_table, {"ID": "r1", "Gene Symbol": "TP53", "Variation": "Mut", "Text": "some text"})
    print("Inserted?", ok)

    print("Inserting duplicate primary r1 (should be False)...")
    ok2 = insert_row(demo_db, demo_table, {"ID": "r1", "Gene Symbol": "BRCA1", "Variation": "Other", "Text": "x"})
    print("Inserted duplicate?", ok2)

    print("Reading table:")
    print(read_table(demo_db, demo_table))

    print("Get single row by primary:")
    print(get_a_row(demo_db, demo_table, "ID", "r1"))

    print("Remove row r1")
    print(remove_row(demo_db, demo_table, "ID", "r1"))

    print("Read table after removal:")
    print(read_table(demo_db, demo_table))
