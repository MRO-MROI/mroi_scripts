import pandas as pd
import matplotlib.pyplot as plt

def readfile(fname):
    """Read in csv file"""
    xg = pd.read_csv(fname, index_col=0, parse_dates=True)
    return xg

def plot_bokeh():
    from bokeh.plotting import figure, output_file, show
    import bokeh.plotting as bk
    #set output file
    output_file("pressure.html", title="XGS600 Pressure")

    # Create a set of tools to use
    TOOLS = "pan,box_zoom,reset,resize,ywheel_zoom"

    # Plot a `line` renderer setting the color, line thickness, title, and legend value.
    p1 = figure(title="XGS600", x_axis_type="datetime", tools=TOOLS, y_axis_type="log", y_range=[10**-7, 10**3], 
        x_axis_label='GM Time', y_axis_label='Torr',plot_width=500)
    p1.line(xg.index, xg['Pressure CNV1'], legend='Pressure CNV1')

    # Plot a `line` renderer setting the color, line thickness, title, and legend value.
    p2 = figure(title="XGS600", x_axis_type="datetime", tools=TOOLS, y_axis_type="log", y_range=[10**-7, 10**3], 
        x_axis_label='GM Time', y_axis_label='Torr',plot_width=500)
    p2.line(xg.index, xg['Pressure IMG1'], legend='Pressure IMG1')

    gp = bk.GridPlot(children=[[p1,p2]])
    show(gp)

def plot_pd():
    dat = readfile('XGS600Log_20150622 17:16:29.csv')
    ax = dat.plot('Pressure CNV1')
    fig = ax.get_figure()
    fig.savefig('xgs.png')

if __name__ == '__main__':
    plot_pd()
