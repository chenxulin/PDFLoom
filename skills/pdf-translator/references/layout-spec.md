# Layout specification

Use this JSON format to place searchable translated text over a preserved source
page background. The builder rasterizes the source at high resolution, covers
only listed regions, and writes the translated text as vector text.

## Coordinate system

- Use PDF points: 72 points = 1 inch.
- Use a top-left origin.
- `x` increases to the right; `y` increases downward.
- Write boxes as `[x0, y0, x1, y1]`.
- Read page dimensions from `source-manifest.json`.
- Use the coordinate-guide images to estimate boxes.
- To convert rendered-image pixels, use:
  `x_pt = x_px × pt_per_px_x` and `y_pt = y_px × pt_per_px_y`.

Include approximately 1–2 points of cover margin around source glyphs without
touching table borders, rules, arrows, stamps, or nearby artwork.

## Complete example

```json
{
  "schema_version": 1,
  "source_pdf": "/absolute/path/source.pdf",
  "source_sha256": "sha256-from-source-manifest",
  "background_dpi": 300,
  "pages": [
    {
      "page": 1,
      "width_pt": 596.16,
      "height_pt": 841.8,
      "translation_complete": true,
      "elements": [
        {
          "id": "p1-title",
          "type": "text",
          "bbox": [92, 74, 510, 121],
          "source": "Original title",
          "text": "Translated Title",
          "font": "bold",
          "font_size": 17,
          "min_font_size": 12,
          "line_height": 1.05,
          "align": "center",
          "valign": "middle",
          "padding": 2,
          "fill": "#FFFFFF",
          "text_color": "#000000",
          "rotation": 0
        }
      ]
    }
  ]
}
```

## Element fields

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Unique stable identifier such as `p2-table-r3-c2`. |
| `type` | Yes | `text` for translated text or `cover` for a non-semantic source mark that must be hidden. |
| `bbox` | Yes | Top-left-coordinate box in PDF points. |
| `source` | Recommended | Visually verified source text for audit and review. |
| `text` | For `text` | Complete target-language translation. |
| `font` | No | `regular`, `bold`, `mono`, or `cjk`; default `regular`. |
| `font_size` | No | Preferred point size; default 10. |
| `min_font_size` | No | Smallest acceptable fitted size; default 6. |
| `line_height` | No | Leading multiplier; default 1.12. |
| `align` | No | `left`, `center`, `right`, or `justify`. |
| `valign` | No | `top`, `middle`, or `bottom`. |
| `padding` | No | Inner padding in points; default 1.5. |
| `fill` | No | Cover color as `#RRGGBB`; default white. Use `null` only when no source text lies underneath. |
| `text_color` | No | Text color as `#RRGGBB`; default black. |
| `border_width` | No | Optional replacement border width in points. |
| `border_color` | No | Replacement border color. |
| `rotation` | No | Clockwise visual rotation: 0, 90, 180, or 270. |

## Page completion

Every source page must have a matching page entry. Templates set
`translation_complete` to `false`. Change it to `true` only after visually
checking that every meaningful source-language text region on that page has a
translated element. Pages containing no translatable text still require visual
review and an explicit `true` value.

For ordinary PDFs with a usable text layer, the template pre-populates source
text blocks and their boxes. Translate and refine those elements rather than
assuming the extracted block geometry is exact. Scanned pages have no reliable
text blocks and require visual element creation from the coordinate guides.

## Granularity

- Use one element per paragraph when it occupies a clean rectangular region.
- Split paragraphs only when columns, illustrations, or irregular wrapping make
  one rectangle unsafe.
- Use one element per table cell; do not cover the table grid.
- Use one element per form label and a separate element for its value.
- Use one element per diagram label; leave connectors and shapes visible.
- Keep headers and footers separate from body text.
- Keep codes and identifiers in `mono` only when the source uses a technical
  monospaced treatment; otherwise preserve the source visual weight.

## Fitting and visual matching

Choose the requested font size from the apparent source size. Let the builder
reduce it only to `min_font_size`. Treat any build overflow as a layout error.
Do not set a very small minimum merely to suppress an error.

For colored or shaded cells, sample or visually approximate the local fill color.
For gradients, textured backgrounds, or lines behind text, use multiple small
cover boxes or reconstruct only the affected local region. Never apply a large
white rectangle over meaningful graphics.

For a bilingual deliverable, do not cover the source text; set `fill` to
`null` and place the translation in a separate region. For a translated-only
deliverable, every meaningful source-language text region must be covered.

## Build report and verification

The build creates `OUTPUT.layout-report.json`. Review every element marked
`shrunk: true`; confirm that the fitted text remains readable and visually
consistent.

The verifier compares source and output pixels outside all translation boxes.
A failure usually means a page-size mismatch, an altered background, or cover
boxes that are missing from the layout specification.
