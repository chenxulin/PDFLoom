---
name: pdf-translator
description: Translate scanned, image-only, or ordinary PDFs into searchable target-language PDFs while preserving the original page geometry, page count, backgrounds, tables, figures, diagrams, logos, fields, and visual hierarchy as closely as possible. Use for reports, contracts, forms, certificates, manuals, technical documents, medical or pharmaceutical records, financial documents, and any request requiring a near-identical or layout-faithful translated PDF.
---

# PDF Translator

Produce a layout-faithful translated PDF. Keep every source page as the visual
background, cover only original-language text regions, and place searchable
translated text into explicit coordinate boxes. Do not reflow the document into
a new template.

## Scope the translation

Determine or infer the source language, target language, audience, register, and
domain. Prioritize a user glossary, approved translation, style guide, official
bilingual name, cited standard, and document-defined terminology in that order.

For legal, regulatory, medical, scientific, financial, technical, academic, or
safety-critical material, read
[references/domain-adaptation.md](references/domain-adaptation.md).

## Prepare the source

Resolve `PDF_TRANSLATOR_DIR` to the directory containing this file. Keep the
source PDF unmodified.

```bash
python3 "$PDF_TRANSLATOR_DIR/scripts/layout_preserving_translation.py" prepare SOURCE.pdf \
  --workdir WORKDIR --dpi 220
python3 "$PDF_TRANSLATOR_DIR/scripts/layout_preserving_translation.py" template \
  WORKDIR/source-manifest.json --output WORKDIR/layout.json
python3 "$PDF_TRANSLATOR_DIR/scripts/layout_preserving_translation.py" guide SOURCE.pdf \
  --output-dir WORKDIR/coordinate-guides --spacing 36
```

Read `source-manifest.json`. Inspect every rendered source page and its
coordinate guide. Treat any extracted text as a hint; visually verify the
source.

## Build the translation layout

Read [references/layout-spec.md](references/layout-spec.md) before authoring
`layout.json`.

Create a page-by-page content ledger. Capture every title, paragraph, list item,
table cell, field, identifier, number, date, amount, unit, equation, footnote,
caption, and meaningful diagram label. Record illegible content explicitly.

Add one layout element for each coherent source text region. For tables, use one
element per cell or visually independent text block. For diagrams and charts,
replace labels individually without covering lines, plotted data, arrows, or
artwork. Preserve logos, stamps, signatures, handwriting, photographs, and
non-text graphics unless the user asks to alter them.

Set each cover box narrowly enough to retain borders and nearby graphics, but
wide enough to hide every source-language glyph. Match the local background
color. Use the source alignment, weight, approximate font size, and rotation.
After visually reconciling every source text region on a page, set that page's
`translation_complete` field to `true`. Never mark an unreviewed page
complete; the builder rejects incomplete pages.

Translate faithfully. Never omit meaning merely to make text fit. If the target
text is longer:

1. Use adjacent whitespace within the same visual region.
2. Adjust wrapping and line spacing.
3. Reduce the font moderately while maintaining readability.
4. Recompose the local region only when the preceding options cannot work.

The build must fail on unresolved text overflow.

## Build and verify

```bash
python3 "$PDF_TRANSLATOR_DIR/scripts/layout_preserving_translation.py" build \
  SOURCE.pdf WORKDIR/layout.json --output TRANSLATED.pdf
python3 "$PDF_TRANSLATOR_DIR/scripts/layout_preserving_translation.py" verify \
  SOURCE.pdf WORKDIR/layout.json TRANSLATED.pdf
python3 "$PDF_TRANSLATOR_DIR/scripts/layout_preserving_translation.py" render \
  TRANSLATED.pdf --output-dir WORKDIR/qa-pages
```

Use `--force` only when intentionally replacing a known output.

The verifier must pass page count, page dimensions, searchable target text,
media-box bounds, and background similarity outside translation boxes. Inspect
every output page side by side with its source page. Check:

- page size, orientation, margins, page numbers, logos, lines, and borders;
- complete removal of source-language text unless bilingual output was requested;
- translation box alignment, font hierarchy, line spacing, and readability;
- every table cell, form field, caption, annotation, and diagram label;
- absence of clipping, collisions, blank covers, or accidental graphic damage.

Rebuild and repeat both mechanical and visual checks after every layout change.

## Fidelity rules

- Preserve page count and page dimensions exactly.
- Preserve names, dates, amounts, currencies, units, equations, references,
  identifiers, versions, and signatures exactly.
- Preserve intent, certainty, obligations, warnings, and technical force.
- Do not silently correct suspicious source values.
- Do not leave source-language text visible unless requested or necessary to
  identify a proper noun.
- Do not cover visual evidence to simplify translation.
- Use red text only when requested or when an illegible value materially affects
  meaning; otherwise disclose uncertainty in the handoff.
- Keep translated text searchable and keep the rasterized source background free
  of its original hidden text layer.

Install dependencies only when missing:

```bash
python3 -m pip install -r "$PDF_TRANSLATOR_DIR/requirements.txt"
```

Deliver the final PDF path, identify missing source pages or attachments, and
state any unresolved ambiguity.
