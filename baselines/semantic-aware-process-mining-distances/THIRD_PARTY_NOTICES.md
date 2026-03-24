# Third-Party Notices

This repository contains original research code as well as code that is
re-used, adapted, or re-implemented from prior work. The project as a whole is
licensed under the MIT License (see `LICENSE`) **except** for portions that are
explicitly attributed to third parties below; those portions remain subject to
their respective licenses and/or terms.

If you believe a license notice is missing or incorrect, please open an issue
or contact the maintainers.

## Included / Adapted Code

- **Levenshtein implementation (PM4Py-derived)**
  - **Path**: `distances/trace_distances/edit_distance/levenshtein/algorithmpm4py.py`
  - **Attribution**: Copyright (c) 2017 Oleg Bulkin
  - **License**: MIT (as stated in the file header)

## Re-used / Re-implemented Methods (by reference)

The following methods are implemented or re-implemented based on prior work and
original author implementations, as documented in `README.md`. These parts are
included for scientific reproducibility. Where code was copied/adapted, we keep
attribution in-source (e.g., in `original_authors.py` modules) and/or in commit
history.

- Bose \& van der Aalst (2009) substitution scores (re-implemented)
- De Koninck et al. (2018) act2vec (based on the authors' released implementation/configs)
- Chiorrini et al. (2022) Embedding Process Structure (re-implemented and optimized)
- Gamallo-Fernández et al. (2023) autoencoder baseline (adapted for this benchmark)

For the most up-to-date pointers to upstream sources, see the table in `README.md`.


