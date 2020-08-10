from datetime import datetime, timedelta
from collections import OrderedDict
from typing import List
from calendar import Calendar


def months_between_dates(start_date: str, end_date: str):
    date_list = [start_date, end_date]
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in date_list]
    month_dates = list(OrderedDict(((start + timedelta(_)).strftime("%Y-%m"), None) for _ in range((end - start).days)).keys())
    return month_dates


def create_year_calendar(year: int) -> List[str]:
    cal = Calendar()
    year_cal = cal.yeardatescalendar(year)
    day_list = list()
    # maravilloso for
    for season_list in year_cal:
        for month_list in season_list:
            for week_list in month_list:
                for day_date in week_list:
                    day_list.append(day_date.strftime('%Y%m%d'))
    return day_list


def create_weekly_year_calendar(year: int) -> List[str]:
    cal = Calendar()
    year_cal = cal.yeardatescalendar(year)
    week_day_list = list()
    # maravilloso for
    for season_list in year_cal:
        for month_list in season_list:
            for week_list in month_list:
                # print(week_list[-1])
                week_day_list.append(week_list[-1].strftime('%Y-%m-%d'))
    return week_day_list
