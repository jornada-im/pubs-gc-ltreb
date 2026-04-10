# pubs-gc-ltreb

Public documentation, analysis, and code products for the global change LTREB projects at the Jornada Basin in southern New Mexico, U.S.A.

**Directories**

├── manuscripts                  - Analysis scripts, figure creation code, etc. for manuscripts
│   ├── schadenfreude            -   Materials for in-review "Schadenfreude" project MS
│   │   ├── SoilMoisture.ipynb   -     Jupyter notebook for a soil moisture analysis
│   │   └── ...
│   └── ...
├── metadata                     - Files with contextual information and metadata
│   ├── lt_plot_trt.csv          -   Plot and treatment table
│   └── ...
├── README.md                    - This file
└── src                          - Common source code for loading or transforming project data
    ├── schadenfreude_helper.py  -   Code for the Schadenfreude project
    └── ...


**Branches**

The repository is organized into branches that have a series of named releases along them. These releases are what get published with Zenodo, usually with the intent of being cited in manuscripts. Current branches:

* `schadenf_submission` is for an in-review manuscript from the "Schadenfreude" project.
