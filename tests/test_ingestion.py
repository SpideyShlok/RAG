import pandas as pd
import ingestion


def test_csv_loads_one_document_per_row(tmp_path):
    df = pd.DataFrame({"Employee": ["Shlok", "Asha"], "Department": ["Engineering", "Sales"], "Salary": [95000, 88000]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)

    docs = ingestion.load_source(str(path))
    assert len(docs) == 2
    assert "Employee: Shlok" in docs[0].page_content
    assert "Department: Engineering" in docs[0].page_content


def test_xlsx_loads_one_document_per_row(tmp_path):
    df = pd.DataFrame({"Employee": ["Shlok", "Asha"], "Salary": [95000, 88000]})
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)

    docs = ingestion.load_source(str(path))
    assert len(docs) == 2
    assert docs[0].metadata["row"] == 0


def test_blank_rows_are_dropped(tmp_path):
    df = pd.DataFrame({"A": ["", "value"], "B": ["", "value2"]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)

    docs = ingestion.load_source(str(path))
    assert len(docs) == 1  # the all-blank row should be dropped
