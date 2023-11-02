import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.1f' % x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", "Year 2010-2011")
df = df_.copy()


def cltv_car(dataframe):
    # Data Prep
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 1]

    def outlier_thresholds(dataframe, variable):
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    num_cols = [i for i in dataframe.columns if dataframe[i].dtype != "O"]
    num_cols.remove("Customer ID")

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # Analiz tarihi
    last_date = dataframe["InvoiceDate"].max()
    gunler_eklenecek = dt.timedelta(days=2)
    analysis_date = last_date + gunler_eklenecek

    #########################
    # Lifetime Veri Yapısının Hazırlanması
    #########################

    # recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
    # T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
    # frequency: tekrar eden toplam satın alma sayısı (frequency>1)
    # monetary: satın alma başına ortalama kazanç

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: ((InvoiceDate.max() - InvoiceDate.min()).days) / 7,
                         lambda InvoiceDate: ((analysis_date - InvoiceDate.min()).days) / 7],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ["recency", "T", "frequency", "monetary"]
    cltv_df = cltv_df[cltv_df["frequency"] > 1]

    ##############################################################
    # BG-NBD Modelinin Kurulması
    ##############################################################
    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    ################################################################
    # Tahmin Sonuçlarının Değerlendirilmesi
    ################################################################

    plot_period_transactions(bgf)
    plt.show()

    ##############################################################
    # GAMMA-GAMMA Modelinin Kurulması
    ##############################################################

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

    ##############################################################
    # BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    ##############################################################

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=6,  # 6 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final.sort_values(by="clv", ascending=False).head(10)

    ##############################################################
    # CLTV'ye Göre Segmentlerin Oluşturulması
    ##############################################################

    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, ["D", "C", "B", "A"])
    cltv_final.groupby("segment").agg(
        {"count", "mean", "sum"})

    return cltv_final

cltv_final = cltv_car(df)

