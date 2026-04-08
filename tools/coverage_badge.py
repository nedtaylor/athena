import os
from pathlib import Path

from bs4 import BeautifulSoup


def resolve_coverage_html() -> Path:
    candidates = []
    override = os.environ.get("COVERAGE_HTML_FILE")
    if override:
        candidates.append(Path(override))

    candidates.extend(
        [
            Path("./build/coverage-ci/coverage/index.html"),
            Path("./build/coverage/index.html"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("Could not locate a coverage HTML report.")


html_file = resolve_coverage_html()

with html_file.open("r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, 'html.parser')

    # Find the "coverage" table
    coverage_table = soup.find('table', {'class': 'coverage'})

    if coverage_table is None:
        raise ValueError("Coverage table was not found in the HTML report.")

    percentage = None
    fallback_percentage = None
    for row in coverage_table.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        if not cells:
            continue

        percent_cells = [cell for cell in cells if cell.endswith('%')]
        if not percent_cells:
            continue

        if fallback_percentage is None:
            fallback_percentage = percent_cells[0]

        if cells[0].lower().rstrip(':') == 'total':
            percentage = percent_cells[0]
            break

    if percentage is None:
        percentage = fallback_percentage

    if percentage is None:
        raise ValueError("Could not extract a percentage value from the coverage report.")
    
print(int(float(percentage.replace("%", ""))))
