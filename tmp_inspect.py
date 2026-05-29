import json
with open(r"c:\Users\linds\repos\Unidos\acs_detailed_estimates.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]

# Print specific cells in full
for i in [18, 20, 22, 24, 26]:
    cell = cells[i]
    src = "".join(cell["source"])
    print(f"=== Cell {i+1} ({cell['cell_type']}) ===")
    print(src)
    print()
