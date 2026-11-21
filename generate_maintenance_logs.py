import os
import random
from datetime import datetime, timedelta
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(101)

assets = [
    "CNC-Mill-01",
    "CNC-Mill-02",
    "Press-Line-A",
    "Hydraulic-Bender-03",
    "Assembly-Cell-B",
]
shifts = ["Morning", "Afternoon", "Night"]
ticket_types = ["Maintenance", "Warranty Claim", "Field Service"]

failure_templates = {
    "Bearing & Spindle Wear": [
        "High spindle vibration and unusual whine on {asset} during {shift} shift; bearing preload check needed.",
        "Elevated spindle temp and noise near front bearing housing on {asset}. Possible wear.",
        "Spindle runout out of spec on {asset}, grinding sound reported by operator.",
        "Bearing rumble detected on {asset} drive shaft; vibration trending upward.",
    ],
    "Hydraulic & Pneumatic Leaks": [
        "Hydraulic fluid leak near manifold seal on {asset} ({shift} shift), pressure dropping.",
        "Air line hiss and pneumatic pressure drop at valve block on {asset}.",
        "Oil puddle under {asset} hydraulic power unit; hose fitting weeping.",
        "Actuator sluggish on {asset} due to pneumatic leak and seal wear.",
    ],
    "Electrical & Sensor Faults": [
        "Proximity sensor fault causing intermittent stop on {asset} during {shift} shift.",
        "PLC input flickering and drive overload alarm triggered on {asset}.",
        "Loose wiring harness at limit switch on {asset}; controller fault code E-14.",
        "Temperature sensor reading erratic on {asset}, power supply voltage dip observed.",
    ],
    "Alignment & Calibration": [
        "Part dimensions drifting out of tolerance on {asset}; axis alignment required.",
        "Tool offset and laser calibration off on {asset} after {shift} shift setup.",
        "Misalignment between press ram and die bed on {asset}, uneven wear pattern.",
        "Feed axis backlash and calibration drift noted on {asset} during QC check.",
    ],
    "Operator & Process Error": [
        "Wrong tool offset entered by operator on {asset} ({shift} shift), caused scrap batch.",
        "Coolant level not topped off before run on {asset}; thermal alarm triggered.",
        "Incorrect cycle recipe selected on {asset}, minor collision with fixture.",
        "Improper clamping torque on {asset} led to part slip mid-cycle.",
    ],
}

severity_map = {
    "Bearing & Spindle Wear": ["High", "Critical", "Medium"],
    "Hydraulic & Pneumatic Leaks": ["High", "Medium", "Critical"],
    "Electrical & Sensor Faults": ["Medium", "High", "Critical"],
    "Alignment & Calibration": ["Medium", "Low", "High"],
    "Operator & Process Error": ["Low", "Medium", "High"],
}

start_date = datetime(2026, 8, 1)
rows = []

for i in range(1200):
    ticket_id = f"MT-{10001 + i}"
    created_at = start_date + timedelta(
        days=random.randint(0, 90),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    asset = random.choice(assets)
    shift = random.choice(shifts)
    ticket_type = random.choices(ticket_types, weights=[0.6, 0.25, 0.15])[0]

    # Inject higher hydraulic/alignment issues on Press-Line-A and Hydraulic-Bender-03
    if asset in ["Press-Line-A", "Hydraulic-Bender-03"]:
        category = random.choices(
            list(failure_templates.keys()),
            weights=[0.15, 0.35, 0.15, 0.25, 0.10],
        )[0]
    else:
        category = random.choices(
            list(failure_templates.keys()),
            weights=[0.28, 0.15, 0.25, 0.17, 0.15],
        )[0]

    note = random.choice(failure_templates[category]).format(asset=asset, shift=shift)
    severity = random.choice(severity_map[category])

    severity_multiplier = {"Low": 0.6, "Medium": 1.0, "High": 1.8, "Critical": 3.0}[severity]
    downtime_hours = round(max(0.2, random.expovariate(1 / 1.5) * severity_multiplier), 2)
    repair_cost_usd = round(max(85.0, random.gauss(420, 160) * severity_multiplier), 2)

    rows.append({
        "Ticket_ID": ticket_id,
        "Date": created_at.strftime("%Y-%m-%d"),
        "Shift": shift,
        "Asset_ID": asset,
        "Ticket_Type": ticket_type,
        "Severity": severity,
        "Downtime_Hours": downtime_hours,
        "Repair_Cost_USD": repair_cost_usd,
        "Technician_Notes": note,
        "Failure_Category": category,
    })

df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
output_csv = os.path.join(DATA_DIR, "maintenance_warranty_logs.csv")
df.to_csv(output_csv, index=False)
print(f"Generated {len(df)} maintenance/warranty tickets at '{output_csv}'.")