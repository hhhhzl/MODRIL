import os
import argparse
import gdown


# These datasets are shared by the authors of https://github.com/clvrai/goal_prox_il
DEMOS = {
    "ant": [
        ("exp", "1aSsk2EowI4nLcL0qp-SbNB5NYlbl2ewI"),
        ("50", "1Kp8ZmQK7rG668-N5yVZDiRkdjf_CcAbd")
    ],
    "hand": [
        ("exp", "1xpeAsvlW7l3lryq8p8AIsgviC2cF6756"),
        ("10000", "1c9M-W56DxT3EhPdvSlW7iTw2RXl84wj3")
    ],
    "maze": [
        ("exp", "1Es2ng45NqR0nfX2QQRclJXPE2mATbYgt"),
        ("25", "1H2-MZ87N_wRVvaSd1_hE79QeKEuYyeHX"),
        ("50", "1DQApVDxQyIlcgfxbgeCpvv2BEgjZeKTc"),
        ("75", "1z5Kd1ipVv_lYC55807R6vqdehzpWQ2j8"),
        ("100", "1Ge8IyjY7xIiz8x-mG3b9O_qIoOooeNyz"),
    ],
    "pick": [
        ("exp", "1JW-UpXBRz3k1zWoAefGTEWy3Oyf_niqH"),
        ("partial3", "1aUJCQk6SRbTyOt9IBMNdNxjb_s-tKpMX"),
    ],
    "push": [
        ("exp", "1oCIAR_FgNVH4VgyJd7g2Tduz4mdj-Rgi"),
        ("partial2", "1laz8DneZLbQYzp7Rcc4PYhs5kquEfI2w"),
    ],
    "walker": [
        ("exp", "1Y8B_UTqW8jlJ41BqbyOzjxJ9iGkp9ULW")
    ],
    "halfcheetah": [
        ("exp", "1Xts1-Zwz7L2MUcn_wKxTK7SKbL_wC-rI")
    ],
    "sine": [
        ("exp", "1g0657mSnedhWYHdPHG4ITs27JRPB9VrU")
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
            # download for experiment only
            if args.v == 'exp':
                if postfix != "exp":
                    continue
                else:
                    target_path = "%s/%s.pt" % (DIR, task)
            else:
                # download for experiment all
                if postfix != "exp":
                    target_path = "%s/%s_%s.pt" % (DIR, task, postfix)
                else:
                    target_path = "%s/%s.pt" % (DIR, task)

            url = "https://drive.google.com/uc?id=" + id
            if os.path.exists(target_path):
                print("%s is already downloaded." % target_path)
            else:
                print("Downloading demo (%s_%s) from %s" % (task, postfix, url))
                gdown.download(url, target_path)