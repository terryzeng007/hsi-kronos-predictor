import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 从Kronos源码导入必要的类
import sys

sys.path.insert(0, './kronos_src')
from model import Kronos, KronosTokenizer, KronosPredictor


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
#model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 3. Prepare Data
df = pd.read_excel("./data/HSI_2.xlsx")
df['timestamps'] = pd.to_datetime(df['date'])
df['pred'] = np.nan

pred_feq = 'half_year'
if pred_feq == 'month':
    lookback = 378
    pred_len = 21
    cycle = int(len(df)/pred_len)-1

    for i in range(cycle):
        lookback = i*pred_len + 378
        print(i, lookback)

        x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df.loc[:lookback-1, 'timestamps']
        y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
        record_date = y_timestamp.iloc[-1].strftime('%Y-%m-%d')

        # 4. Make Prediction
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

        df.loc[lookback:lookback+pred_len-1, 'pred_close'] = pred_df['close'].values
        df.loc[lookback:lookback + pred_len - 1, 'pred_high'] = pred_df['high'].values
        df.loc[lookback:lookback+pred_len-1, 'pred_low'] = pred_df['low'].values
        df_export = df.loc[lookback:lookback+pred_len-1]

        # export
        df_export.to_csv('D:/Git_Project/agent2_graph/predict_rolling/' + pred_feq + '/' +str(lookback) + '_' + record_date + '.csv')

elif pred_feq == 'week':
        lookback = 95
        pred_len = 5
        cycle = int(len(df) / pred_len) - 1

        for i in range(cycle):
            lookback = i * pred_len + 95
            print(i, lookback)

            x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = df.loc[:lookback - 1, 'timestamps']
            y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']
            record_date = y_timestamp.iloc[-1].strftime('%Y-%m-%d')

            # 4. Make Prediction
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=True
            )

            df.loc[lookback:lookback + pred_len - 1, 'pred_close'] = pred_df['close'].values
            df.loc[lookback:lookback + pred_len - 1, 'pred_high'] = pred_df['high'].values
            df.loc[lookback:lookback + pred_len - 1, 'pred_low'] = pred_df['low'].values
            df_export = df.loc[lookback:lookback + pred_len - 1]

            # export
            df_export.to_csv('D:/Git_Project/agent2_graph/predict_rolling/' + pred_feq + '/'  + record_date + '.csv')

elif pred_feq == 'two_month':
    lookback = 400
    pred_len = 42
    cycle = int(len(df) / pred_len) - 1

    for i in range(cycle):
        lookback = i * pred_len + 400
        print(i, lookback)

        x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df.loc[:lookback - 1, 'timestamps']
        y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']
        record_date = y_timestamp.iloc[-1].strftime('%Y-%m-%d')

        # 4. Make Prediction
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

        df.loc[lookback:lookback + pred_len - 1, 'pred_close'] = pred_df['close'].values
        df.loc[lookback:lookback + pred_len - 1, 'pred_high'] = pred_df['high'].values
        df.loc[lookback:lookback + pred_len - 1, 'pred_low'] = pred_df['low'].values
        df_export = df.loc[lookback:lookback + pred_len - 1]

        # export
        df_export.to_csv('D:/Git_Project/agent2_graph/predict_rolling/' + pred_feq + '/' + record_date + '.csv')

elif pred_feq == 'half_year':
    lookback = 400
    pred_len = 120
    cycle = int(len(df) / pred_len) - 1

    for i in range(cycle):
        lookback = i * pred_len + 400
        print(i, lookback)

        x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df.loc[:lookback - 1, 'timestamps']
        y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']
        record_date = y_timestamp.iloc[-1].strftime('%Y-%m-%d')

        # 4. Make Prediction
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

        df.loc[lookback:lookback + pred_len - 1, 'pred_close'] = pred_df['close'].values
        df.loc[lookback:lookback + pred_len - 1, 'pred_high'] = pred_df['high'].values
        df.loc[lookback:lookback + pred_len - 1, 'pred_low'] = pred_df['low'].values
        df_export = df.loc[lookback:lookback + pred_len - 1]

        # export
        df_export.to_csv('D:/Git_Project/agent2_graph/predict_rolling/' + pred_feq + '/' + record_date + '.csv')

    #
    #
        #