import sys
import os
import argparse
import gdown


# These datasets are shared by the authors of https://github.com/clvrai/goal_prox_il
DEMOS = {
    "ant": [
        ("", "1ST9_V_ddV4mdbhNnidx3r7BNabHki33m"),
        ("50", "1ST9_V_ddV4mdbhNnidx3r7BNabHki33m")
    ],
    "hand": [
        ("", "1ST9_V_ddV4mdbhNnidx3r7BNabHki33m"),
        ("10000", "1NsZ8FrTIyVvxEiAyTRDtHyzcKfCZNlZu")
    ],
    "maze": [
        ("", ""),
        ("25", "1Il1SWb0nX8RT796izf-YkqvYyb3ls8yO"),
        ("50", "1xfrhsFQEY__pCYe-6xYmPSkPdyrw-reD"),
        ("75", "1A4F3eammJaLWiV2HxAh9lKiij4dDgj6h"),
        ("100", "1Eocidtv_BUwmXQlVgF17rkRerxOX-mvM"),
    ],
    "pick": [
        ("", ""),
        ("partial3", "1xrAw_ic0DOjfBSl6P6btP4oVGsXFmKNB"),
    ],
    "push": [
        ("", ""),
        ("partial2", "1kV48YTLdYO3SYN8OQk6KNWa12aCjqJcB"),
    ],
    "walker": [
        ("", "1kV48YTLdYO3SYN8OQk6KNWa12aCjqJcB")
    ],
    "halfcheetah": [
        ("", "")
    ],
    "sine": [
        ("", "1kV48YTLdYO3SYN8OQk6KNWa12aCjqJcB")
    ]
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="../expert_datasets")
    parser.add_argument("--v", type=str, default="exp", choices=['exp', 'full'])
    args = parser.parse_args()

    DIR = args.dir

    tasks = ["ant", "hand", "maze", "pick", "push", "walker", "halfcheetah", "sine"]

    os.makedirs(DIR, exist_ok=True)

    for task in tasks:
        for postfix, id in DEMOS[task]:
            if args.v == 'exp': # download for experiment only
                if postfix != "":
                    continue
                else:
                    target_path = "%s/%s.pt" % (DIR, task)
            else:
                if postfix != "": # download for experiment all
                    target_path = "%s/%s_%s.pt" % (DIR, task, postfix)
                else:
                    target_path = "%s/%s.pt" % (DIR, task)

            url = "https://drive.google.com/uc?id=" + id
            if os.path.exists(target_path):
                print("%s is already downloaded." % target_path)
            else:
                print("Downloading demo (%s_%s) from %s" % (task, postfix, url))
                gdown.download(url, target_path)