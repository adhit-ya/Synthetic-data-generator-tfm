"""Small dependency-free Excel preview writer.

The project is intentionally Python-only and already exports complete datasets
as CSV/Parquet/Torch. This module writes a compact `.xlsx` preview workbook so
humans can open generated worlds directly in Excel without requiring openpyxl or
xlsxwriter at runtime.
"""

from __future__ import annotations

import math
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


INVALID_SHEET_CHARS = set("[]:*?/\\")


def export_excel_preview(
    world_id: str,
    tables: Dict[str, pd.DataFrame],
    output_path: Path,
    max_rows_per_sheet: int = 500,
) -> None:
    """Write one Excel workbook with a summary and table preview sheets."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheets: List[tuple[str, pd.DataFrame]] = [
        ("Summary", _build_summary(world_id, tables)),
    ]
    for table_name, table in tables.items():
        if table.empty:
            continue
        preview = table.head(max_rows_per_sheet).copy()
        sheets.append((_safe_sheet_name(table_name), preview))
    _write_xlsx(output_path, sheets)


def _build_summary(world_id: str, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table_name, table in tables.items():
        rows.append(
            {
                "world_id": world_id,
                "table": table_name,
                "rows": len(table),
                "columns": table.shape[1],
                "average_missing_rate": 0.0 if table.empty else table.isna().mean().mean(),
                "target_columns": ", ".join(c for c in table.columns if c.endswith("_target")),
            }
        )
    return pd.DataFrame(rows)


def _safe_sheet_name(name: str) -> str:
    cleaned = "".join("_" if char in INVALID_SHEET_CHARS else char for char in name)
    return cleaned[:31] or "Sheet"


def _write_xlsx(path: Path, sheets: Sequence[tuple[str, pd.DataFrame]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types(len(sheets)))
        archive.writestr("_rels/.rels", _root_relationships())
        archive.writestr("docProps/core.xml", _core_properties())
        archive.writestr("docProps/app.xml", _app_properties([name for name, _ in sheets]))
        archive.writestr("xl/workbook.xml", _workbook_xml([name for name, _ in sheets]))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_relationships(len(sheets)))
        archive.writestr("xl/styles.xml", _styles_xml())
        for index, (_, frame) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(frame))


def _worksheet_xml(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [headers] + frame.replace({pd.NaT: None}).where(pd.notna(frame), None).values.tolist()
    dimension = f"A1:{_column_letter(max(1, len(headers)))}{max(1, len(rows))}"
    xml_rows = []
    for row_index, row in enumerate(rows, start=1):
        xml_cells = []
        for col_index, value in enumerate(row, start=1):
            style = ' s="1"' if row_index == 1 else ""
            xml_cells.append(_cell_xml(row_index, col_index, value, style))
        xml_rows.append(f'<row r="{row_index}">{"".join(xml_cells)}</row>')
    col_defs = "".join(
        f'<col min="{i}" max="{i}" width="{_column_width(header)}" customWidth="1"/>'
        for i, header in enumerate(headers, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="{dimension}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft"/>
    </sheetView>
  </sheetViews>
  <cols>{col_defs}</cols>
  <sheetData>{"".join(xml_rows)}</sheetData>
  <autoFilter ref="{dimension}"/>
</worksheet>"""


def _cell_xml(row: int, col: int, value: object, style: str = "") -> str:
    ref = f"{_column_letter(col)}{row}"
    if value is None:
        return f'<c r="{ref}"{style}/>'
    if isinstance(value, (datetime, pd.Timestamp)):
        text = escape(value.isoformat())
        return f'<c r="{ref}" t="inlineStr"{style}><is><t>{text}</t></is></c>'
    if isinstance(value, date):
        text = escape(value.isoformat())
        return f'<c r="{ref}" t="inlineStr"{style}><is><t>{text}</t></is></c>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}" t="b"{style}><v>{1 if bool(value) else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        return f'<c r="{ref}"{style}><v>{float(value)}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"{style}><is><t>{text}</t></is></c>'


def _column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _column_width(header: str) -> int:
    return min(max(len(header) + 3, 12), 36)


def _content_types(sheet_count: int) -> str:
    sheet_overrides = "\n".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  {sheet_overrides}
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def _root_relationships() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _workbook_xml(sheet_names: Sequence[str]) -> str:
    sheets_xml = "\n".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{sheets_xml}</sheets>
</workbook>"""


def _workbook_relationships(sheet_count: int) -> str:
    relationships = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, sheet_count + 1)
    )
    relationships += (
        f'\n<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {relationships}
</Relationships>"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E79"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  </cellXfs>
</styleSheet>"""


def _core_properties() -> str:
    created = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>synthetic_enterprise_generator</dc:creator>
  <cp:lastModifiedBy>synthetic_enterprise_generator</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>"""


def _app_properties(sheet_names: Sequence[str]) -> str:
    names = "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name in sheet_names)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>synthetic_enterprise_generator</Application>
  <TitlesOfParts>
    <vt:vector size="{len(sheet_names)}" baseType="lpstr">{names}</vt:vector>
  </TitlesOfParts>
</Properties>"""
