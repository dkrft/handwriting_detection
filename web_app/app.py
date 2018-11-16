# Load the packages

from flask import Flask, render_template, request

import cv2

# Connect the app
app = Flask(__name__)


@app.route('/')
def homepage():
    """Render home page"""
    return render_template('home.html')


# @app.route('/graph', methods=["GET", "POST"])
# def graph():
#     script, div = "", ""
#     ticker = request.form['ticker']
#     search = ""
#     ticks = ""

#     try:
#         data = quandl.get("WIKI/%s" % ticker)
#         message = ""
#         ticks = ticker
#         search = "https://www.google.com/search?q=%s" % (ticker)

#         oldest, newest = data.index.min(), data.index.max()
#         old_spec = np.datetime64(request.form["start"])
#         new_spec = np.datetime64(request.form["end"])

#         if new_spec <= old_spec:
#             message = "The supplied date range was given incorrectly. \
#             Start date (%s) must be a later date than end date(%s). \
#             Please submit a new query." % (old_spec, new_spec)
#             return render_template('graph.html', message=message,
#                                    script=script, div=div)

#         mask = (data.index >= old_spec) & (data.index <= new_spec)
#         data = data[mask]

#         dt = (new_spec - old_spec)
#         option = bool(int(request.form["yes_no"]))

#         if dt > np.timedelta64(2 * 365, 'D') and dt <= np.timedelta64(3 * 365, 'D')\
#                 and option:
#             data = data.resample('W').mean()
#         if dt > np.timedelta64(3 * 365, 'D') and option:
#             data = data.resample('M').mean()

#         if data.shape[0] > 0:
#             # create plot, as have data
#             p = get_plot(data, ticker, request)
#             script, div = components(p)

#             if old_spec < np.datetime64(oldest):
#                 message = "Please note that the supplied data range was %s to %s. \
#             The data was available in a subset of this range from %s to %s" % (
#                     old_spec, new_spec, np.datetime64(oldest, 'D'),
#                     np.datetime64(newest, 'D'))
#         else:
#             message = "%s exists but no data in chosen date range of %s to %s.\
#              \nPlease select something between %s to %s." % (
#                 ticker, old_spec, new_spec, oldest, newest)
#     except NotFoundError:
#         message = "%s is not found in the Quandl/WIKI API" % ticker
#     except:
#         message = "Undefined error associated with implementation."

#     return render_template('graph.html', message=message, ticker=ticks,
#                            search=search, script=script, div=div)

if __name__ == '__main__':
    app.run(debug=True)  # Set to false when deploying
