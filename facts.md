Facts for the thesis data section

Size and mix: 250 synthetic tickets, 46% resolve, 41% escalate, 13% ask_info. A model that always does one thing scores at most about 46%.
Hard cases: 18 injection, 24 no-runbook, 21 always-escalate and 15 needs-approval tickets.
How it was built: generated from runbooks, then checked by an audit model from a different family, plus a second human-style review of 94 tickets. Most hard tickets were reinstated, 3 are flagged as weak, and every change is in final_v2_changes.csv.
Limitations: synthetic and cleaner than real tickets, and the resolve versus ask_info boundary is fuzzy, which is why acceptable_actions exists.