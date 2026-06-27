import nbformat
import uuid

nb = nbformat.read('Q5_Individual_AlternativeTokenization.ipynb', as_version=4)

cell_source = r'''print("=" * 60)
print("STEP 7: TOKEN-BOUNDARY VISUALISATION -- THE 'open-class' CASE")
print("=" * 60)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Take a small window of tokens around 'open-class' from each method's
# output and render each token as its own box. This is the kind of
# token-boundary view used by tools such as the Hugging Face tokenizer
# visualiser -- it shows directly WHERE the three tokenizers agree and
# where they disagree, rather than just a total count.

nltk_idx = tokens_nltk.index("open-class")
default_idx = tokens_spacy_default.index("open")
custom_idx = tokens_spacy_custom.index("open-class")

CORRECT = "#55A868"  # green -- 'open-class' kept as ONE token
PROBLEM = "#C44E52"  # red   -- 'open-class' split into separate tokens
NORMAL = "#DCE6F1"   # light blue -- ordinary token

rows = [
    ("NLTK word_tokenize\n(Q1, group)",
     tokens_nltk[nltk_idx - 3:nltk_idx + 4], {3}, CORRECT),
    ("spaCy (default)",
     tokens_spacy_default[default_idx - 3:default_idx + 5], {3, 4, 5}, PROBLEM),
    ("spaCy (custom rule)",
     tokens_spacy_custom[custom_idx - 3:custom_idx + 4], {3}, CORRECT),
]

fig, ax = plt.subplots(figsize=(13, 4))

for row_i, (label, tokens, highlight, hl_color) in enumerate(rows):
    y = len(rows) - 1 - row_i
    x = 0.0
    for i, tok in enumerate(tokens):
        color = hl_color if i in highlight else NORMAL
        width = 0.42 * len(tok) + 0.4
        ax.add_patch(plt.Rectangle((x, y - 0.35), width, 0.7,
                                    facecolor=color, edgecolor="white"))
        ax.text(x + width / 2, y, tok, ha="center", va="center", fontsize=10)
        x += width + 0.1
    ax.text(-0.3, y, label, ha="right", va="center", fontsize=10, fontweight="bold")

ax.set_xlim(-5.5, 26)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.axis("off")
ax.set_title("Token Boundaries Around 'open-class' — NLTK vs spaCy (default) vs spaCy (custom rule)",
              fontsize=12)

legend_patches = [
    mpatches.Patch(facecolor=CORRECT, label="'open-class' kept as ONE token"),
    mpatches.Patch(facecolor=PROBLEM, label="'open-class' SPLIT into separate tokens"),
    mpatches.Patch(facecolor=NORMAL, label="other tokens"),
]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("Q5_token_highlight.png", bbox_inches="tight")
plt.show()

print("\nChart saved as 'Q5_token_highlight.png'")
print("=" * 60)
'''

cell = nbformat.v4.new_code_cell(source=cell_source)
cell.id = uuid.uuid4().hex[:8]

# Insert after Step 6 (bar chart, index 7), before the Justification cell
nb.cells.insert(8, cell)

nbformat.validate(nb)
nbformat.write(nb, 'Q5_Individual_AlternativeTokenization.ipynb')
print("Done. Total cells:", len(nb.cells))
