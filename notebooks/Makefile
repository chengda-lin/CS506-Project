PYTHON ?= python3
NOTEBOOK ?= eda.ipynb
OUT_DIR ?= outputs

.PHONY: install run test clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter nbconvert

run:
	$(PYTHON) -m jupyter nbconvert --execute --to notebook "$(NOTEBOOK)" --output "$(NOTEBOOK)"
	
test:
	$(PYTHON) -c "import pandas, numpy, matplotlib, seaborn, sklearn, imblearn; print('Environment check passed.')"

clean:
	rm -rf $(OUT_DIR) .ipynb_checkpoints