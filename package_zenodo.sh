#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
zip -r shenzhen_aging_data.zip data/
echo "Zenodo package created: shenzhen_aging/shenzhen_aging_data.zip"
echo "Size: $(du -h shenzhen_aging_data.zip | cut -f1)"
