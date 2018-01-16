import os

from csxdata.visual import Scatter2D

from homyd import Dataset, LearningTable
from homyd.features import transformation_factory


suite = Dataset(LearningTable.from_xlsx(
    os.path.expanduser("~/SciProjects/Project_Fruits/adat.xlsx"),
    labels=["FAMILIA"], paramset=["DH1", "DH2", "D13C"]), transformation=transformation_factory("pca", 2)
)
Scatter2D(*suite.table("learning"), title="HoMyD TestRun", axlabels=["PC1", "PC2"]).split_scatter(show=True)
