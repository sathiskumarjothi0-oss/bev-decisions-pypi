"""Ready-made question sets, worded like the question families Bev is tested on.

    python examples/07_presets.py
"""

import os

import bev

print(bev.presets.names())
model = bev.load(os.environ.get("BEV_MODEL") or bev.DEFAULT_MODEL)
log = "sshd[2211]: Failed password for root from 203.0.113.7 port 52211 (x47 in 2 minutes)"
for name, a in model.decide(log, bev.presets.get("security_log")).items():
    print(name, bev.value(a), round(bev.confidence(a), 3))
invoice = "Invoice INV-2291\nLaptop stand $89.00\nMonitor $1,240.00\nDocking station $310.50\nTotal: $1,639.50"
print(model.decide(invoice, {"over_limit": bev.presets.over_limit("$1,500")}))
