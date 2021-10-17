import numpy as np
from reprod_log import ReprodDiffHelper


diff_helper = ReprodDiffHelper()
info1 = diff_helper.load_info("./torch_data.npy")
info2 = diff_helper.load_info("./paddle_data.npy")

diff_helper.compare_info(info1, info2)
diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff.txt")