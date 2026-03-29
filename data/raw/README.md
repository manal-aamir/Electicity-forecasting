# Raw data

Place the UCI dataset here, or let the pipeline download it:

- Expected file after download: `household_power_consumption.txt`
- Source: [UCI — Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

Run from the project root:

```bash
python run_hw_xgb_xai.py --scales hourly --target global_active_power
```

(`*.txt` and `*.zip` in this folder are ignored by Git because they are large.)
