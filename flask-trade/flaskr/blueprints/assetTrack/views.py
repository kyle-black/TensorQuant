from flask import Blueprint, render_template




asset_track_BP = Blueprint('asset_track', __name__, url_prefix= '/asset_track')


@asset_track_BP.route('/EURUSD')
def index():
    return render_template('asset/eurusd.html')