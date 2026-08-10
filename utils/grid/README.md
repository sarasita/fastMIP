# Grid

This folder documents the common latitude–longitude grid used in fastMIP Phase 2.

## Remapping

Model output is remapped onto a common lat–lon grid using the CDO conservative second-order remapping operator (`remapcon2`).
This ensures consistency across modelling teams while conserving spatially integrated quantities.

A reference card for the CDO operators is provided in:
- `cdo_refcard.pdf`

The grid specification itself is described in:
- `g025.txt`

## Land–sea mask

MESMER is land-specific. Consequently, a land–sea mask is applied after remapping to retain land grid cells only.

It is still under discussion whether analyses and data provision should:
- focus exclusively on land-area quantities, or
- retain both land and ocean grid cells after remapping.

This choice becomes particularly relevant when computing regional aggregates, where consistency between grid definition, masking, and regional boundaries is required.
