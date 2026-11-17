"""Add (or remove) an `aliases` list in every runbook: how employees tend to word the problem.

  python src/retrieval/add_aliases.py            # adds aliases; first run backs up data/runbooks to data/runbooks_backup_before_aliases
  python src/retrieval/add_aliases.py --remove   # strips them again, to compare retrieval with and without
Then rebuild the index (the runbook fingerprint changes) and re-run evaluate_retrieval.py.

The aliases were written from general knowledge of how people describe IT problems, for all 40 runbooks alike.
They were not copied from ticket text. Because the dev misses were seen before they were written, report dev results
after this step as tuned on dev, and keep the test split untouched until the final run.
"""
import argparse
import glob
import os
import shutil

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ALIASES = {
    "RB-PWD-01": ["forgot my password", "cannot remember my password", "do not remember my login password", "need a password reset link", "password not working, need a new one", "reset my password"],
    "RB-PWD-02": ["password expired", "system says I must change my password", "told to update my password", "password change required at login", "password has run out", "password is due for renewal"],
    "RB-PWD-03": ["did not get the verification code", "text message code not coming", "authenticator code not received", "no sms for login", "waiting for the one time passcode", "code never arrives on my phone"],
    "RB-PWD-04": ["got a new phone and cannot approve login", "lost my phone with the authenticator app", "changed my phone number", "old phone was reset, now stuck at approval", "authenticator app needs to move to a new device", "cannot approve the sign-in prompt anymore"],
    "RB-ACC-01": ["account is locked", "locked out after wrong password", "too many failed attempts", "locked out of my laptop login", "account disabled after sign-in attempts", "please unlock my account"],
    "RB-ACC-02": ["need access to a shared mailbox", "add me to the team group", "cannot see the shared inbox", "join the distribution list", "access to our team mailbox", "added to a group but it is not showing"],
    "RB-ACC-03": ["just started", "new joiner", "first week at the company", "joined recently", "account was not ready when I arrived", "cannot reach any of the standard tools", "nothing set up for me on my first day"],
    "RB-ACC-04": ["need access to an application", "please give me access to the app", "request access to a business system", "add me as a user", "new access request for a tool", "want permission to use a system"],
    "RB-VPN-01": ["vpn not connecting", "cannot connect to the vpn", "vpn fails when I try to connect", "secure connection fails", "stuck on connecting", "cannot reach internal systems from home"],
    "RB-VPN-02": ["vpn keeps dropping", "vpn disconnects every few minutes", "vpn is very slow", "connection drops while on the vpn", "vpn lags", "keeps getting kicked off the vpn"],
    "RB-VPN-03": ["vpn trust error", "certificate error when connecting", "vpn profile is broken", "not trusted warning on the vpn", "profile missing or corrupt", "security certificate problem on connection"],
    "RB-VPN-04": ["need the vpn client", "vpn app asks me to update", "install the vpn software", "vpn client missing on my laptop", "vpn update will not install", "vpn application needs a new version"],
    "RB-NET-01": ["wifi not connecting", "cannot join the office wifi", "laptop will not connect to wifi", "wifi shows no network", "wireless not working", "cannot find the corporate wifi"],
    "RB-NET-02": ["no internet on cable", "ethernet not working", "docking station has no network", "wired connection shows no internet", "network cable plugged in but nothing", "desk network port not working"],
    "RB-NET-03": ["internet very slow", "network is crawling", "pages load slowly", "downloads are slow", "everything lags on the network", "slow connection today"],
    "RB-EML-01": ["outlook not syncing", "outlook shows disconnected", "emails not updating", "outlook says working offline", "new mail not coming in", "outlook stuck loading"],
    "RB-EML-02": ["mailbox is full", "storage full cannot send email", "mailbox quota exceeded", "inbox over its limit", "cannot receive mail, no space", "need to clear mailbox space"],
    "RB-EML-03": ["meeting invite not showing", "calendar not updating", "invites missing from my calendar", "accepted meeting does not appear", "calendar out of sync", "cannot see meetings I was invited to"],
    "RB-EML-04": ["email bounced back", "message not delivered", "undeliverable email", "delivery failed error", "sent mail returned with an error", "recipient did not get my email"],
    "RB-SFT-01": ["need software installed", "please install an application", "install a program on my laptop", "want a tool from the approved list", "software request", "need an app on my machine"],
    "RB-SFT-02": ["unlicensed product", "license not activated", "software says not licensed", "activation error", "license key not working", "office shows an unlicensed banner"],
    "RB-SFT-03": ["update failed", "windows update stuck", "update keeps restarting", "software update will not finish", "stuck on updating", "update error loop"],
    "RB-HW-01": ["laptop is slow", "computer freezing", "machine takes forever to start", "laptop running slowly", "everything is sluggish on my pc", "laptop hangs when I open files"],
    "RB-HW-02": ["laptop not charging", "battery drains fast", "charger not working", "battery dies quickly", "laptop will not hold a charge", "charging light does not come on"],
    "RB-HW-03": ["cracked screen", "broken keyboard", "laptop damaged", "dropped my laptop", "keys not working", "screen broken or flickering", "liquid spilled on the laptop"],
    "RB-PRT-01": ["cannot print", "print job stuck in the queue", "printer not printing", "documents will not print", "printer shows offline", "nothing comes out of the printer"],
    "RB-PRT-02": ["add a printer", "printer not in my list", "set up the floor printer", "need the printer installed on my laptop", "map the network printer", "cannot find the printer to add"],
    "RB-PRT-03": ["scan to email not working", "scanner not sending documents", "scanned document never arrives", "scan to folder failing", "copier will not email scans", "scanner emails are missing"],
    "RB-COL-01": ["mic not working in teams", "camera not working in a meeting", "no sound in the call", "others cannot hear me", "webcam shows a black screen", "audio cutting out on calls"],
    "RB-COL-02": ["cannot join the meeting", "meeting link not working", "stuck at joining", "error when joining a teams call", "join button does nothing", "cannot enter the video call"],
    "RB-COL-03": ["teams keeps asking me to sign in", "sign in loop", "app asks for login again and again", "cannot stay signed in to teams", "login screen keeps coming back", "collaboration app will not let me in"],
    "RB-DRV-01": ["shared drive missing", "mapped drive disappeared", "cannot see the network drive", "drive not showing in explorer", "z drive is gone", "lost connection to the shared folder drive"],
    "RB-DRV-02": ["access denied to a folder", "permission denied on the shared folder", "cannot open the folder, no permission", "need access to a folder", "not authorized to open the directory", "you do not have permission message"],
    "RB-DRV-03": ["deleted a file by mistake", "overwrote my document", "need an older version of a file", "recover a deleted file", "file disappeared from the shared folder", "restore a previous version"],
    "RB-APP-01": ["cannot log in to the app", "internal tool login fails", "login keeps failing on our system", "stuck on the login screen", "single sign on error in the application", "app says invalid credentials"],
    "RB-APP-02": ["application is slow", "app keeps timing out", "system hangs when saving", "screen freezes then an error", "internal system very slow", "page takes forever to load in the tool"],
    "RB-APP-03": ["report will not run", "export to excel fails", "download of the report fails", "cannot export to pdf", "report errors out", "data export stuck"],
    "RB-SEC-01": ["suspicious email", "phishing email", "clicked a link in a strange email", "email asking for my password", "fake message pretending to be hr", "entered my password on a strange site"],
    "RB-SEC-02": ["lost my laptop", "laptop stolen", "cannot find my company laptop", "left my phone in a taxi", "device is missing", "someone took my work laptop"],
    "RB-SEC-03": ["unexpected sign in alert", "login prompts I did not start", "mfa prompts keep coming", "someone may be using my account", "strange login notification", "approval requests I never made"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runbooks", default=f"{ROOT}/data/runbooks")
    ap.add_argument("--remove", action="store_true", help="strip the aliases again")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.runbooks, "*.yaml")))
    if not files:
        raise SystemExit(f"no runbook files found in {a.runbooks}")
    backup = a.runbooks.rstrip("/\\") + "_backup_before_aliases"
    if not a.remove and not os.path.exists(backup):
        shutil.copytree(a.runbooks, backup)
        print(f"backup written to {backup}")

    changed, missing = 0, []
    for path in files:
        with open(path, encoding="utf-8") as f:
            rb = yaml.safe_load(f)
        rid = rb["id"]
        if a.remove:
            if rb.pop("aliases", None) is not None:
                changed += 1
            else:
                continue
        else:
            if rid not in ALIASES:
                missing.append(rid)
                continue
            if rb.get("aliases") == ALIASES[rid]:
                continue
            rb["aliases"] = ALIASES[rid]
            changed += 1
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(rb, f, sort_keys=False, allow_unicode=True, width=1000)
    have = {yaml.safe_load(open(p, encoding="utf-8"))["id"] for p in files}
    unused = sorted(set(ALIASES) - have)
    print(f"{'removed aliases from' if a.remove else 'set aliases on'} {changed} of {len(files)} runbooks")
    if missing:
        print(f"WARNING: no aliases defined for: {missing}")
    if unused and not a.remove:
        print(f"note: aliases defined for runbooks that do not exist here: {unused}")
    print("Now rebuild the index:  python src/retrieval/build_index.py --embedder fastembed")


if __name__ == "__main__":
    main()
