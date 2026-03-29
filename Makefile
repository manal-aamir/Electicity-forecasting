PYTHON ?= python
PAPER_TEX ?= IEEE_paper_draft.tex
PAPER_PDF ?= IEEE_paper_draft.pdf

.PHONY: help install run run-hourly run-all-scales run-appliance run-ablation run-lag clean paper paper-open

help:
	@echo "Available targets:"
	@echo "  install         - Install Python dependencies"
	@echo "  run             - Default experiment (hourly, global_active_power)"
	@echo "  run-hourly      - Hourly global forecast with XAI outputs"
	@echo "  run-all-scales  - Global target for hourly/daily/weekly/monthly/quarterly"
	@echo "  run-appliance   - Hourly appliance-level forecasting"
	@echo "  run-ablation    - Hourly run with feature drop-one ablation"
	@echo "  run-lag         - Hourly run with added target lag features"
	@echo "  paper           - Compile IEEE paper PDF from $(PAPER_TEX)"
	@echo "  paper-open      - Compile and open PDF (macOS)"
	@echo "  clean           - Remove LaTeX auxiliary files"

install:
	$(PYTHON) -m pip install -r requirements.txt

run: run-hourly

run-hourly:
	$(PYTHON) run_hw_xgb_xai.py --scales hourly --target global_active_power

run-all-scales:
	$(PYTHON) run_hw_xgb_xai.py --scales hourly daily weekly monthly quarterly --target global_active_power

run-appliance:
	$(PYTHON) run_hw_xgb_xai.py --scales hourly --target sub_metering_1 sub_metering_2 sub_metering_3

run-ablation:
	$(PYTHON) run_hw_xgb_xai.py --scales hourly --target global_active_power --ablation --ablation_top_n 10

run-lag:
	$(PYTHON) run_hw_xgb_xai.py --scales hourly --target global_active_power --target_lags 1 24 168

paper:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode "$(PAPER_TEX)"; \
	else \
		pdflatex -interaction=nonstopmode "$(PAPER_TEX)"; \
		pdflatex -interaction=nonstopmode "$(PAPER_TEX)"; \
	fi

paper-open: paper
	open "$(PAPER_PDF)"

clean:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc *.synctex.gz
