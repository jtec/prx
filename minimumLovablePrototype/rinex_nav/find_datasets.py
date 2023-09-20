import os
import bs4
import pandas as pd
from pathlib import Path
from earthdata_download.download import download_url

from helpers import convert_size_to_bytes


def main():
    survey = pd.DataFrame()
    for week in range(2276, 2000, -1):
        url = f"https://cddis.nasa.gov/archive/gnss/products/{week}/"
        username = "janboltingToulouse"
        password = "Quaxolotl1&"
        scraped_file = Path("./cddis_download") / (str(week) + ".html")
        os.makedirs(scraped_file.parent, exist_ok=True)
        response = download_url(
            url,
            username,
            password,
            local_filename=str(scraped_file),
            skip_existing=True,
            user_agent=None,
        )
        with open(scraped_file, "r") as f:
            html = f.read()
        soup = bs4.BeautifulSoup(html, "lxml")
        table_rows = soup.find_all("div", {"class": "archiveItemTextContainer"})
        dataframe_rows = []
        for row in table_rows:
            dataframe_rows.append(
                {
                    "file": row.find("a", {"class": "archiveItemText"}).text,
                    "file_size_bytes": convert_size_to_bytes(
                        row.find("span", {"class": "fileInfo"}).text.split(" ")[-1]
                    ),
                }
            )
        df = pd.DataFrame(dataframe_rows)
        df = df[df["file"].str.contains("01D_05M_ORB.SP3.gz")]
        df.sort_values(by="file_size_bytes", ascending=False, inplace=True)
        max_this_week = df.iloc[0, :]
        max_this_week["week"] = week
        print(
            f"Largest 1-day sp3 file in week {week}: {max_this_week['file_size_bytes']} bytes"
        )
        survey = pd.concat([survey, max_this_week.to_frame().transpose()])
        print(f"Max so far: {survey['file_size_bytes'].max()} bytes")
        survey.to_csv("survey.csv")
        pass


if __name__ == "__main__":
    main()
