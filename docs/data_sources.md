# Data Sources

## SNB Data Portal (data.snb.ch)

### Cube: `snbgwdzid`
**SNB Monetary Policy Instruments – Daily**

URL: `https://data.snb.ch/api/cube/snbgwdzid/data/csv/en`

| Column | Description | Available from |
|--------|-------------|----------------|
| LZ     | SNB policy rate | Jun 2019 |
| ZIG    | Sight-deposit rate above threshold | Jan 2015 |
| ZIGBL  | Sight-deposit rate below threshold | Sep 2022 (0 % before) |
| FREI   | Threshold factor | Nov 2019 (20 before) |
| ZIABP  | Discount in basis points | Sep 2022 |
| SARON  | SARON overnight fixing | Aug 2009 |
| ENG    | Special rate (Sondersatz) | May 2004 |

**Notes:**
- ZIGBL was 0 % (exempt) during the negative-rate era (Jan 2015 – Jun 2022).  
  The SNB did not record this in the data; we back-fill with 0.
- FREI was 20 from system introduction (Jan 2015) until Oct 2019.  
  We back-fill with 20.
- The SNB policy rate (LZ) was formally introduced on 13 June 2019 as a  
  replacement for the 3-month CHF LIBOR target range. The effective rate  
  before that date equals ZIG (= -0.75 %).

### Current Rates Excel (www.snb.ch)
URL: `https://www.snb.ch/public/rates/interestRates.xlsx`  
Contains the most recent ~250 trading days; used for cross-validation.

## Policy Decisions
Compiled manually from SNB press releases and monetary policy assessments.  
See `input/processed/policy_decisions.csv` for full table.
