from typing import Optional

from src.analysis.spectre_utils import SpectreSamplingMetrics
from src.g2gcompress import Graph2GraphPairCompress


class SocialSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule, g2gc: Optional[Graph2GraphPairCompress]=None):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre'],
                         g2gc=g2gc)
