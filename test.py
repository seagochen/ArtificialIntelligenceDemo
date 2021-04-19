from Cv.CvTools.DiagnosticTool import load_image
from Cv.CvTools.DiagnosticTool import hist
from Cv.CvTools.ImageWithChartPlot import PltDiagramPlotCache


def analysis_image_hists(filepath: str):
    img = load_image(filepath)
    img_hist = hist(img)
    for bin in img_hist:
        print(bin)
    return img, img_hist


if __name__ == "__main__":
    chart = PltDiagramPlotCache()
    img, img_hist = analysis_image_hists("./Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/Fig0503 ("
                                         "original_pattern).tif")
    chart.add_image(img, "origin")
    chart.add_chart(img_hist, "hist")

    chart.plots(1, 2)