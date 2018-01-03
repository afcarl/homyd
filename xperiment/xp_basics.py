import os

from homyd.frame.learningtable import LearningTable
from homyd.features.transformation import transformation_factory

from csxdata.visual import Scatter2D


lt = LearningTable.from_xlsx(os.path.expanduser("~/SciProjects/Project_Fruits/adat.xlsx"),
                             labels="FAMILIA", paramset=["DH1", "DH2", "D13C"])
lt.dropna(inplace=True)
lt.set_transformation(transformation_factory("pca", 2), apply=True)

scat = Scatter2D(lt.X, lt.Y, title="TestEm", axlabels=lt.paramset)
scat.split_scatter(show=True)
