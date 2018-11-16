# Load the packages
import quandl
from quandl.errors.quandl_error import NotFoundError

from flask import Flask, render_template, request

from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.plotting import figure

import numpy as np

# Connect the app
app = Flask(__name__)


def get_plot(df, title, form):
    """Create plot based on what checked or not"""

    p1 = figure(x_axis_type="datetime", width=700, height=500,
                sizing_mode='scale_both', title="Stock Prices for %s" % title)

    if request.form.get('adj_open'):
        if request.form.get('open'):
            p1.circle(df.index, df['Adj. Open'], color="#e74c3c", size=10,
                      legend='Adj. open', alpha=0.3)
        else:
            p1.line(df.index, df['Adj. Open'], color="#e74c3c", line_width=3,
                    legend='Adj. open')

    if request.form.get('adj_close'):
        if request.form.get('close'):
            p1.circle(df.index, df['Adj. Close'], color="#2ecc71", size=10,
                      legend='Adj. close', alpha=0.3)
        else:
            p1.line(df.index, df['Adj. Close'], color="#2ecc71", line_width=3,
                    legend='Adj. close')

    if request.form.get('open'):
        p1.line(df.index, df['Open'], color="#9b59b6", line_width=3,
                legend='Open')

    if request.form.get('close'):
        p1.line(df.index, df['Close'], color="#3498db", line_width=3,
                legend='Close')

    p1.xaxis.axis_label = 'Date'
    p1.xaxis.axis_label_text_font_size = "20pt"
    p1.xaxis.major_label_text_font_size = "16pt"

    p1.yaxis.axis_label = 'Price'
    p1.yaxis.axis_label_text_font_size = "20pt"
    p1.yaxis.major_label_text_font_size = "16pt"

    p1.title.text_font_size = '30pt'
    p1.legend.label_text_font_size = "16pt"
    p1.legend.location = "bottom_right"

    p1.add_tools(HoverTool())
    return(p1)


@app.route('/')
def homepage():
    """Render home page"""
    return render_template('home.html')

quandl.ApiConfig.api_key = "ovneRiWzLYcVUsqxWZbv"


@app.route('/graph', methods=["GET", "POST"])
def graph():
    script, div = "", ""
    ticker = request.form['ticker']
    search = ""
    ticks = ""

    try:
        data = quandl.get("WIKI/%s" % ticker)
        message = ""
        ticks = ticker
        search = "https://www.google.com/search?q=%s" % (ticker)

        oldest, newest = data.index.min(), data.index.max()
        old_spec = np.datetime64(request.form["start"])
        new_spec = np.datetime64(request.form["end"])

        if new_spec <= old_spec:
            message = "The supplied date range was given incorrectly. \
            Start date (%s) must be a later date than end date(%s). \
            Please submit a new query." % (old_spec, new_spec)
            return render_template('graph.html', message=message,
                                   script=script, div=div)

        mask = (data.index >= old_spec) & (data.index <= new_spec)
        data = data[mask]

        dt = (new_spec - old_spec)
        option = bool(int(request.form["yes_no"]))

        if dt > np.timedelta64(2 * 365, 'D') and dt <= np.timedelta64(3 * 365, 'D')\
                and option:
            data = data.resample('W').mean()
        if dt > np.timedelta64(3 * 365, 'D') and option:
            data = data.resample('M').mean()

        if data.shape[0] > 0:
            # create plot, as have data
            p = get_plot(data, ticker, request)
            script, div = components(p)

            if old_spec < np.datetime64(oldest):
                message = "Please note that the supplied data range was %s to %s. \
            The data was available in a subset of this range from %s to %s" % (
                    old_spec, new_spec, np.datetime64(oldest, 'D'),
                    np.datetime64(newest, 'D'))
        else:
            message = "%s exists but no data in chosen date range of %s to %s.\
             \nPlease select something between %s to %s." % (
                ticker, old_spec, new_spec, oldest, newest)
    except NotFoundError:
        message = "%s is not found in the Quandl/WIKI API" % ticker
    except:
        message = "Undefined error associated with implementation."

    return render_template('graph.html', message=message, ticker=ticks,
                           search=search, script=script, div=div)

if __name__ == '__main__':
    app.run(debug=True)  # Set to false when deploying
