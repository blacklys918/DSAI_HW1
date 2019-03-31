{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import linear_model\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.feature_selection import f_regression\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20170102</td>\n",
       "      <td>25.7</td>\n",
       "      <td>21.2</td>\n",
       "      <td>23.2</td>\n",
       "      <td>24.7</td>\n",
       "      <td>25.8</td>\n",
       "      <td>1</td>\n",
       "      <td>24483</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20170103</td>\n",
       "      <td>21.6</td>\n",
       "      <td>22.2</td>\n",
       "      <td>25.5</td>\n",
       "      <td>24.7</td>\n",
       "      <td>24.2</td>\n",
       "      <td>1</td>\n",
       "      <td>28131</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>20170104</td>\n",
       "      <td>26.2</td>\n",
       "      <td>24.6</td>\n",
       "      <td>24.9</td>\n",
       "      <td>24.8</td>\n",
       "      <td>25.4</td>\n",
       "      <td>0</td>\n",
       "      <td>28485</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>20170105</td>\n",
       "      <td>25.6</td>\n",
       "      <td>22.6</td>\n",
       "      <td>26.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>26.4</td>\n",
       "      <td>0</td>\n",
       "      <td>28336</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20170106</td>\n",
       "      <td>23.9</td>\n",
       "      <td>20.5</td>\n",
       "      <td>24.4</td>\n",
       "      <td>25.7</td>\n",
       "      <td>28.3</td>\n",
       "      <td>0</td>\n",
       "      <td>28002</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          0     1     2     3     4     5  6      7\n",
       "0  20170102  25.7  21.2  23.2  24.7  25.8  1  24483\n",
       "1  20170103  21.6  22.2  25.5  24.7  24.2  1  28131\n",
       "2  20170104  26.2  24.6  24.9  24.8  25.4  0  28485\n",
       "3  20170105  25.6  22.6  26.0  25.0  26.4  0  28336\n",
       "4  20170106  23.9  20.5  24.4  25.7  28.3  0  28002"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"training1.csv\" , delim_whitespace=True, header=None)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "features =[\n",
    "    \"Date\",\n",
    "    \"TaipeiTemp\",\n",
    "    \"TaoyuanTemp\",\n",
    "    \"TaichungTemp\",\n",
    "    \"TainanTemp\",\n",
    "    \"KaohsingTemp\",\n",
    "    \"vacation\"\n",
    "]\n",
    "\n",
    "target= \"peak(MW)\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20170102</td>\n",
       "      <td>25.7</td>\n",
       "      <td>21.2</td>\n",
       "      <td>23.2</td>\n",
       "      <td>24.7</td>\n",
       "      <td>25.8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20170103</td>\n",
       "      <td>21.6</td>\n",
       "      <td>22.2</td>\n",
       "      <td>25.5</td>\n",
       "      <td>24.7</td>\n",
       "      <td>24.2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>20170104</td>\n",
       "      <td>26.2</td>\n",
       "      <td>24.6</td>\n",
       "      <td>24.9</td>\n",
       "      <td>24.8</td>\n",
       "      <td>25.4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>20170105</td>\n",
       "      <td>25.6</td>\n",
       "      <td>22.6</td>\n",
       "      <td>26.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>26.4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20170106</td>\n",
       "      <td>23.9</td>\n",
       "      <td>20.5</td>\n",
       "      <td>24.4</td>\n",
       "      <td>25.7</td>\n",
       "      <td>28.3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          0     1     2     3     4     5  6\n",
       "0  20170102  25.7  21.2  23.2  24.7  25.8  1\n",
       "1  20170103  21.6  22.2  25.5  24.7  24.2  1\n",
       "2  20170104  26.2  24.6  24.9  24.8  25.4  0\n",
       "3  20170105  25.6  22.6  26.0  25.0  26.4  0\n",
       "4  20170106  23.9  20.5  24.4  25.7  28.3  0"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "targets=df[7]\n",
    "df.drop(columns=[7], inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>TaipeiTemp</th>\n",
       "      <th>TaoyuanTemp</th>\n",
       "      <th>TaichungTemp</th>\n",
       "      <th>TainanTemp</th>\n",
       "      <th>KaohsingTemp</th>\n",
       "      <th>vacation</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20170102</td>\n",
       "      <td>25.7</td>\n",
       "      <td>21.2</td>\n",
       "      <td>23.2</td>\n",
       "      <td>24.7</td>\n",
       "      <td>25.8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20170103</td>\n",
       "      <td>21.6</td>\n",
       "      <td>22.2</td>\n",
       "      <td>25.5</td>\n",
       "      <td>24.7</td>\n",
       "      <td>24.2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>20170104</td>\n",
       "      <td>26.2</td>\n",
       "      <td>24.6</td>\n",
       "      <td>24.9</td>\n",
       "      <td>24.8</td>\n",
       "      <td>25.4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>20170105</td>\n",
       "      <td>25.6</td>\n",
       "      <td>22.6</td>\n",
       "      <td>26.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>26.4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20170106</td>\n",
       "      <td>23.9</td>\n",
       "      <td>20.5</td>\n",
       "      <td>24.4</td>\n",
       "      <td>25.7</td>\n",
       "      <td>28.3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Date  TaipeiTemp  TaoyuanTemp  TaichungTemp  TainanTemp  KaohsingTemp  \\\n",
       "0  20170102        25.7         21.2          23.2        24.7          25.8   \n",
       "1  20170103        21.6         22.2          25.5        24.7          24.2   \n",
       "2  20170104        26.2         24.6          24.9        24.8          25.4   \n",
       "3  20170105        25.6         22.6          26.0        25.0          26.4   \n",
       "4  20170106        23.9         20.5          24.4        25.7          28.3   \n",
       "\n",
       "   vacation  \n",
       "0         1  \n",
       "1         1  \n",
       "2         0  \n",
       "3         0  \n",
       "4         0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns=features\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>TaipeiTemp</th>\n",
       "      <th>TaoyuanTemp</th>\n",
       "      <th>TaichungTemp</th>\n",
       "      <th>TainanTemp</th>\n",
       "      <th>KaohsingTemp</th>\n",
       "      <th>vacation</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>25.7</td>\n",
       "      <td>21.2</td>\n",
       "      <td>23.2</td>\n",
       "      <td>24.7</td>\n",
       "      <td>25.8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>21.6</td>\n",
       "      <td>22.2</td>\n",
       "      <td>25.5</td>\n",
       "      <td>24.7</td>\n",
       "      <td>24.2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>26.2</td>\n",
       "      <td>24.6</td>\n",
       "      <td>24.9</td>\n",
       "      <td>24.8</td>\n",
       "      <td>25.4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>25.6</td>\n",
       "      <td>22.6</td>\n",
       "      <td>26.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>26.4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>23.9</td>\n",
       "      <td>20.5</td>\n",
       "      <td>24.4</td>\n",
       "      <td>25.7</td>\n",
       "      <td>28.3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   TaipeiTemp  TaoyuanTemp  TaichungTemp  TainanTemp  KaohsingTemp  vacation\n",
       "0        25.7         21.2          23.2        24.7          25.8         1\n",
       "1        21.6         22.2          25.5        24.7          24.2         1\n",
       "2        26.2         24.6          24.9        24.8          25.4         0\n",
       "3        25.6         22.6          26.0        25.0          26.4         0\n",
       "4        23.9         20.5          24.4        25.7          28.3         0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.drop(columns=[\"Date\"], inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "features_vectors=df.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test= train_test_split(features_vectors, targets ,test_size=0.1,random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Slope: -27.65954775931606\n",
      "Intercept: 18316.64857593886\n",
      "Socre Training:  0.8314931934847327\n",
      "Socre Testing:  0.8303594944057319\n"
     ]
    }
   ],
   "source": [
    "reg=linear_model.LinearRegression()\n",
    "reg.fit(X_train, Y_train)\n",
    "print(\"Slope:\", reg.coef_[0])\n",
    "print(\"Intercept:\", reg.intercept_)\n",
    "print(\"Socre Training: \", reg.score(X_train, Y_train))\n",
    "print(\"Socre Testing: \", reg.score(X_test, Y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F values: [628.15927399 805.36025023 438.07873785 453.64340538 483.07023849\n",
      " 366.18687176]\n",
      "p values: [1.08128923e-099 7.49900534e-119 4.53107946e-076 3.79187064e-078\n",
      " 5.33109800e-082 4.31931056e-066]\n"
     ]
    }
   ],
   "source": [
    "F_values, p_values =f_regression(X_train ,Y_train)\n",
    "print(\"F values:\", F_values)\n",
    "print(\"p values:\", p_values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('TaipeiTemp', 628.1592739923104),\n",
       " ('TaoyuanTemp', 805.3602502319184),\n",
       " ('TaichungTemp', 438.07873785311256),\n",
       " ('TainanTemp', 453.64340537956485),\n",
       " ('KaohsingTemp', 483.0702384919562),\n",
       " ('vacation', 366.18687175937777)]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "features_and_f_values= list(zip(df.columns,F_values))\n",
    "features_and_f_values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "features_num_seq = range(1, len(features))\n",
    "result_test_scores = list()\n",
    "result_training_scores = list()\n",
    "for num in features_num_seq:\n",
    "    num_of_choosen_features = num\n",
    "    selected_features = [\n",
    "        feature_and_f_value[0]\n",
    "        for feature_and_f_value in features_and_f_values[:num_of_choosen_features]\n",
    "    ]\n",
    "    \n",
    "    features_vectors = df[selected_features].values\n",
    "    X_train, X_test, Y_train, Y_test = train_test_split(features_vectors, targets, test_size=0.1, random_state=1)\n",
    "    \n",
    "    reg = linear_model.LinearRegression()\n",
    "    reg.fit(X_train, Y_train)\n",
    "    \n",
    "    result_training_scores.append(reg.score(X_train, Y_train))\n",
    "    result_test_scores.append(reg.score(X_test, Y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYsAAAELCAYAAAAoUKpTAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xl8FfW5+PHPkz2EkAAJKCSBqMguoBG1uOAG6LWK1iLY9lrbW7ppW/uTW733aq237aW1e69daGvtcglaUURFgYooKpiEfZMtQDb2ELbs5zy/P2YCJyEhATKZ5Jzn/Xqd15n5zvYM0e9z5vud+Y6oKsYYY8yZRPkdgDHGmM7PkoUxxphWWbIwxhjTKksWxhhjWmXJwhhjTKssWRhjjGmVp8lCRCaJyBYR2S4ijzWzPEtE3hGR1SKyTkRud8sHikiViKxxP7/zMk5jjDFnJl49ZyEi0cBW4FagBMgHpqnqppB1ZgGrVfW3IjIMWKCqA0VkIPC6qo7wJDhjjDFnxcsri7HAdlUtVNVaYA5wV5N1FOjhTqcAZR7GY4wx5hx5mSz6A8Uh8yVuWaingM+KSAmwAHg4ZFm22zz1rohc52GcxhhjWhHj4b6lmbKmbV7TgOdV9acicg3wNxEZAewBslT1kIhcAcwTkeGqerTRAUSmA9MBkpKSrhgyZEj7n4UxxoSxlStXHlTV9NbW8zJZlACZIfMZnN7M9EVgEoCqLheRBCBNVfcDNW75ShHZAVwKFIRurKqzgFkAOTk5WlDQaLExxphWiMjutqznZTNUPjBIRLJFJA6YCsxvsk4RcDOAiAwFEoADIpLudpAjIhcBg4BCD2M1xhhzBp5dWahqvYg8BCwEooHnVHWjiDwNFKjqfOD/AX8QkUdwmqg+r6oqItcDT4tIPRAAvqKq5V7Faowx5sw8u3W2o1kzlDHGnD0RWamqOa2t52Wfhe/q6uooKSmhurra71A8l5CQQEZGBrGxsX6HYowJQ2GdLEpKSkhOTmbgwIGINHdzVnhQVQ4dOkRJSQnZ2dl+h2OMCUNhnSyqq6vDPlEAiAi9e/fmwIEDfodijOlA81aX8szCLZRVVNEvNZEZEwczeUzTx9naR9gPJBjuiaJBpJynMcYxb3Upj7+8ntqKMubEPU1txR4ef3k981aXenK8sE8WfquoqOA3v/nNWW93++23U1FR4UFExphw8MzCLVTVBfhmzFyulC18I+ZlquoCPLNwiyfHC+tmqM6gIVl87Wtfa1QeCASIjo5ucbsFCxZ4HZoxpgt7u2oKCQl1J+c/F/NPPhfzT6qrYoGD7X48u7IIMW91KeNmLiH7sTcYN3NJu1zOPfbYY+zYsYPRo0dz5ZVXcuONN3L//fczcuRIACZPnswVV1zB8OHDmTVr1sntBg4cyMGDB9m1axdDhw7lS1/6EsOHD2fChAlUVVWdd1zGmK5tSsLveL3+KhqefqjSOF6pH8enE37vyfEsWbga2v9KK6pQoLSiql3a/2bOnMnFF1/MmjVreOaZZ8jLy+MHP/gBmzY5I7U/99xzrFy5koKCAn71q19x6NCh0/axbds2vv71r7Nx40ZSU1OZO3fuecVkjOn6vjDpGnrJUUSgVmOIp47qqG58cdLVnhwvYpqhvvfaRjaVHW1x+eqiCmoDwUZlVXUB/v2ldeTmFTW7zbB+PfjuJ4efVRxjx45tdHvrr371K1555RUAiouL2bZtG7179260TXZ2NqNHjwbgiiuuYNeuXWd1TGNM+PnExb2JlhKOaiJTa5/gi93e44Y+9fTz6G6oiEkWrWmaKForP1dJSUknp5cuXco///lPli9fTrdu3Rg/fnyzDxDGx8efnI6OjrZmKGMMb32Qz79GHaN87KMsuP3rwNc9PV7EJIvWrgDGzVxCacXplXD/1ERe+PI153zc5ORkjh071uyyI0eO0LNnT7p168bHH3/MihUrzvk4xpjIEQwq1atmA9Drmn/tkGNan4VrxsTBJMY2vjspMTaaGRMHn9d+e/fuzbhx4xgxYgQzZsxotGzSpEnU19dz2WWX8cQTT3D11d60NRpjwssH2w9wS80SDva+EnoO6JBjRsyVRWsannr04mnI2bNnN1seHx/Pm2++2eyyhn6JtLQ0NmzYcLL80UcfPe94jDFd24r33mJG1F7qrvnPDjumJYsQk8f09+xReWOMaQ8HjtXQb/er1MXGEzticocd15qhjDGmC5mXv4M7oj6k+uJ/gYQeHXZcu7IwxpguIhhUSj56hRSphKs/16HHtisLY4zpIlYUHuK6ysVUJfSF7Bs69NiWLIwxpot47cM1jI9eS+yYqRDV8thyXrBmKGOM6QIOHa+h+7Z5xEQHYcz9HX58T68sRGSSiGwRke0i8lgzy7NE5B0RWS0i60Tk9pBlj7vbbRGRiV7G6aVzHaIc4Be/+AWVlZXtHJExpiuau6qEyfIe1emjoM+QDj++Z8lCRKKBZ4HbgGHANBEZ1mS1/wJeVNUxwFTgN+62w9z54cAk4Dfu/rocSxbGmPOlqny0fBnDo3aTkPNZX2LwshlqLLBdVQsBRGQOcBewKWQdBRru/UoBytzpu4A5qloD7BSR7e7+lnsYr+PYXnjpQbj3eUjue967Cx2i/NZbb6VPnz68+OKL1NTUcPfdd/O9732PEydOMGXKFEpKSggEAjzxxBPs27ePsrIybrzxRtLS0njnnXfO/9yMMV3SisJyrjq2iGBsDFEjPuVLDF4mi/5Acch8CXBVk3WeAhaJyMNAEnBLyLahAyWVuGWNiMh0YDpAVlZWuwTNuz+GohXw7o/gjp+d9+5mzpzJhg0bWLNmDYsWLeKll14iLy8PVeXOO+/kvffe48CBA/Tr14833ngDcMaMSklJ4Wc/+xnvvPMOaWlp5x2HMabreuGjQv4r5n100ERI6t36Bh7wMlk091JobTI/DXheVX8qItcAfxOREW3cFlWdBcwCyMnJOW15I28+BnvXt7y86ANOvkUEoOBPzkcEssY1v80FI+G2mWc8bKhFixaxaNEixowZA8Dx48fZtm0b1113HY8++ijf+c53uOOOO7juuuvavE9jTHgrP1HL8U2LSYs54kvHdgMvk0UJkBkyn8GpZqYGX8Tpk0BVl4tIApDWxm3bV78r4fBOqDoEGgSJgm69oWd269u2kary+OOP8+Uvf/m0ZStXrmTBggU8/vjjTJgwgSeffLLdjmuM6bpeXlXCXfIu9Qk9iRk0wbc4vEwW+cAgEckGSnE6rJumxSLgZuB5ERkKJAAHgPnAbBH5GdAPGATknVc0bbkCeO0RWPU8xCRAoBaG3nneTVGhQ5RPnDiRJ554gs985jN0796d0tJSYmNjqa+vp1evXnz2s5+le/fuPP/88422tWYoYyKTqvLqR5uYG72SmMsehJg432LxLFmoar2IPAQsBKKB51R1o4g8DRSo6nzg/wF/EJFHcJqZPq+qCmwUkRdxOsPrga+rasCrWE86sR+ueBByHoSCP8Pxfee9y9Ahym+77Tbuv/9+rrnGeT9G9+7d+fvf/8727duZMWMGUVFRxMbG8tvf/haA6dOnc9ttt3HhhRdaB7cxEShvZznDD79DXGwdjJrqayyieuam/q4iJydHCwoKGpVt3ryZoUOH+hRRx4u08zUm3H1rzmr+9eMvM7q3EvVQntOH2s5EZKWq5rS2ng33YYwxnVBFZS3rN6zhcrYQNeZ+TxLF2bBkYYwxndDLq0q5k/dQBEZO8TscSxbGGNPZqCpzPtrF1LgPkIvGQ4r/L2UL+2QRLn0yrYmU8zQmEqzcfZjUgyvpG9wHo6b5HQ4Q5skiISGBQ4cOhX1FqqocOnSIhIQEv0MxxrSD2XlFTI17H41LgqF3+B0OEOZDlGdkZFBSUsKBAwf8DsVzCQkJZGRk+B2GMeY8Hams4+11u/hh3EfIsHsgLsnvkIAwTxaxsbFkZ7ffE9jGGOO1V1aXcEMwj4Rgpe/PVoQK62YoY4zpSlSV3LxiHkxaDilZMKCFcel8YMnCGGM6iVVFFVTs282oujUw6j6I6jxVdOeJxBhjIlxuXhFT4j4kimCnuQuqQVj3WRhjTFdxpKqO19eV8m63D6HPVdD7Yr9DasSuLIwxphN4dU0pl9TvoG/Nrk7Vsd3AkoUxxvhMVZn9URHTUz6C6HgYfo/fIZ3GkoUxxvhsTXEFO/YeZkLgfRhyOySm+h3SaSxZGGOMz3LzipgYt46EusOdrmO7gXVwG2OMj45V1/Ha2j38o2ceBNLh4pv9DqlZdmVhjDE+enVNGfF1FQw7vtwZijy6c/6Gt2RhjDE+Odmx3XM1UcE6GN05m6DA42QhIpNEZIuIbBeRx5pZ/nMRWeN+topIRciyQMiy+V7GaYwxflhfeoRNe44yJe596DsSLhjpd0gt8ux6R0SigWeBW4ESIF9E5qvqpoZ1VPWRkPUfBsaE7KJKVUd7FZ8xxvgtN6+IYbF7STuyAa76gd/hnJGXVxZjge2qWqiqtcAc4K4zrD8NyPUwHmOM6TSO19Tz6poyHu27EiQaRn7a75DOyMtk0R8oDpkvcctOIyIDgGxgSUhxgogUiMgKEZnsXZjGGNPx5q8po7q2jmurlsAlN0NyX79DOiMvu92lmbKWXlk3FXhJVQMhZVmqWiYiFwFLRGS9qu5odACR6cB0gKysrPaI2RhjOkRuXhH39d5J3Ik9MOqHfofTKi+vLEqAzJD5DKCshXWn0qQJSlXL3O9CYCmN+zMa1pmlqjmqmpOent4eMRtjjOfWlxxhfekR/i3lI4hPgcG3+x1Sq7xMFvnAIBHJFpE4nIRw2l1NIjIY6AksDynrKSLx7nQaMA7Y1HRbY4zpinLzi+gdW8NFB5bAiLshNsHvkFrlWbJQ1XrgIWAhsBl4UVU3isjTInJnyKrTgDmqGtpENRQoEJG1wDvAzNC7qIwxpqs6UVPPq6tLmZG5FamrhFH3+x1Sm3j6qKCqLgAWNCl7ssn8U81s9yHQeW84NsaYc/Ta2jJO1Aa4PbgUel0EmWP9DqlN7AluY4zpQLl5RVyXXkWPvcudQQOluXuBOh9LFsYY00E2lB5hbckRvtVnlVNw2X3+BnQWLFkYY0wHmZNfRHyMMLr8TRhwLfQc4HdIbWbJwhhjOkBlbT3zVpfxtYvLiT5c2KkHDWyOJQtjjOkAr6/dw/GaeqYlfAgxiTD0ztY36kQsWRhjTAfIzS9iaHoc6btfh6GfhIQefod0VixZGGOMxzbvOcrqogoeHbgTqT7S5ZqgwJKFMcZ4bk5eEXExUVxXuRiS+0H2DX6HdNYsWRhjjIeqagO8vLqUKUPiiCt8Gy6bAlHRfod11jrny16NMSZMvLF+D8eq6/lS6jrQgPMgXhdkVxbGGOOh3LwiLkpPIqvkVeg3BvoM8Tukc2LJwhhjPLJl7zFW7j7MQ0Orkb3ru8yggc2xZGGMMR7JzSsiLjqK24LvQlQsjPiU3yGdM0sWxhjjgeq6AC+vKuH24Wkkbn4JLp0ISb39DuucWbIwxhgPLFi/h6PV9Xw5owhO7O+yHdsNLFkYY4wHcvOKyE5LYsi+1yGxFwya4HdI58WShTHGtLNt+46Rv+swD4xJRT5+A0beCzFxfod1XixZGGNMO8vNKyY2Wrg3sQACNTBqqt8hnTdPk4WITBKRLSKyXUQea2b5z0VkjfvZKiIVIcseEJFt7ucBL+M0xpj2Ul0XYO6qEiYMv4Dum/8BaYOh3+V+h3XePHuCW0SigWeBW4ESIF9E5qvqpoZ1VPWRkPUfBsa4072A7wI5gAIr3W0PexWvMca0h7c27OVIVR1fGKrw6gq45aku8+rUM/HyymIssF1VC1W1FpgD3HWG9acBue70RGCxqpa7CWIxMMnDWI0xpl3MzitiQO9uXH74LUBg5BS/Q2oXXiaL/kBxyHyJW3YaERkAZANLznZbY4zpLHYcOE7eznKm5mQg6+bAReMhJTyqLi+TRXPXXdrCulOBl1Q1cDbbish0ESkQkYIDBw6cY5jGGNM+5uQVERMlTLugFCqKuvyzFaG8TBYlQGbIfAZQ1sK6UznVBNXmbVV1lqrmqGpOenr6eYZrjDHnrqY+wEsrS5gwvC+pW/8Bcd1h6B1+h9VuvEwW+cAgEckWkTichDC/6UoiMhjoCSwPKV4ITBCRniLSE5jglhljTKe0cOM+DlfW8ZnL02HjqzBsMsQl+R1Wu/HsbihVrReRh3Aq+WjgOVXdKCJPAwWq2pA4pgFzVFVDti0Xkf/GSTgAT6tquVexGmPM+cr9qIjMXolcU7sCao+FxbMVoTx9+ZGqLgAWNCl7ssn8Uy1s+xzwnGfBGWNMOyk8cJzlhYeYMXEwUet+DSlZMGCc32G1K3uC2xhjztML+cXERAn3DY6GwqUw6j6ICq/qNbzOxhhjOlhNfYB/rCzhlqF9SSt8FTQYVndBNbBkYYwx52Hxpn2Un6hl2thMWJsLmVdB74v9DqvdWbIwxpjzkJtXRP/URK5LKoUDH4ddx3YDSxbGGHOOdh08wQfbDzFtbCZR63IhOh6G3+N3WJ6wZGGMMedoTn4x0VHCp8f0hQ0vwZDbITHV77A8YcnCGGPOQW19kJdWFnPzkD703bcMKg+FZcd2A0sWxhhzDv65eR8Hj9cy7aosWDMbktLh4pv9DsszliyMMeYcNHRsX98/GrYudIYij/b0OWdfWbIwxpizVFxeybJtB5mSk0n0ppchWAejw7cJCs4iWYjItSLyoDudLiLZ3oVljDGd15z8IqIEplyZ4Txb0XckXDDS77A81aZkISLfBb4DPO4WxQJ/9yooY4zprOoCQV4sKOGmIX24sLYYSleG7bMVodp6ZXE3cCdwAkBVy4Bkr4IyxpjO6u3N+zlwrIZpY7Ng7WyQaBj5ab/D8lxbk0WtO4S4AohI+AzSbowxZyE3r4gLUxK44ZJesO5FuORmSO7rd1iea2uyeFFEfg+kisiXgH8Cf/AuLGOM6XyKyyt5b9sBpuRkElP0PhwtDetnK0K16T4vVf2JiNwKHAUGA0+q6mJPIzPGmE7mxYJiBJhyZSYs+SnEp8Dg2/0Oq0O0mixEJBpYqKq3AJYgjDERqT4Q5IX8YsYP7kP/xHrYPB8umwKxCX6H1iFabYZS1QBQKSIpHRCPMcZ0Sks+3s/+ho7tTfOhrhJG3e93WB2mrY8bVgPrRWQx7h1RAKr6DU+iMsaYTiY3r4i+PeK5cXA6/C0Xel0EmWP9DqvDtLWD+w3gCeA9YGXI54xEZJKIbBGR7SLyWAvrTBGRTSKyUURmh5QHRGSN+5nfxjiNMabdlVZUsXTrAe7LySTmWAnsWuZ0bIv4HVqHaWsH919EJA641C3aoqp1Z9rG7et4FrgVKAHyRWS+qm4KWWcQzoN+41T1sIj0CdlFlaqOPotzMcYYT7yQXwy4HdvrnnUKL7vPx4g6Xluf4B4PbMOp/H8DbBWR61vZbCywXVULVbUWmAPc1WSdLwHPquphAFXdfxaxG2OM5+oDQV7ML+aGS9PJSE2ENbkw4FroOcDv0DpUW5uhfgpMUNUbVPV6YCLw81a26Q8Uh8yXuGWhLgUuFZEPRGSFiEwKWZYgIgVu+eQ2xmmMMe1q6ZYD7D1a7XRsl+RD+Y6wHzSwOW3t4I5V1S0NM6q6VURiW9mmucY8beb4g4DxQAawTERGqGoFkKWqZSJyEbBERNar6o5GBxCZDkwHyMrKauOpGGNM2+XmFdEnOZ6bhvSBN38EMYkw9E6/w+pwbb2yKBCRP4nIePfzB1rv4C4BMkPmM4CyZtZ5VVXrVHUnsAUneTSMP4WqFgJLgTFND6Cqs1Q1R1Vz0tPT23gqxhjTNnuOVPHOlv1MyckkNlgLG+bC0E9CQg+/Q+twbU0WXwU2At8AvglsAr7Syjb5wCARyXY7x6cCTe9qmgfcCCAiaTjNUoUi0lNE4kPKx7nHNMaYDvNifgkK3HdlJmx9E6qPRGQTFLS9GSoG+KWq/gxO3ukUf6YNVLVeRB4CFgLRwHOqulFEngYKVHW+u2yCiGwCAsAMVT0kIp8Afi8iQZyENjP0LipjjPFaIKi8kF/EtZekkdmrG7w1B5L7QfYNfofmi7Ymi7eBW4Dj7nwisAj4xJk2UtUFwIImZU+GTCvwbfcTus6HQHi/ScQY06m9t/UAZUeqeeKOYXB8P2xbDJ94GKKi/Q7NF21thkpQ1YZEgTvdzZuQjDHGf7PzikjrHs8tw/rC+pdAAxEzwmxz2posTojI5Q0zIpIDVHkTkjHG+GvvkWqWfLyfT+dkEBsd5bzkqN8Y6DPE79B809ZmqG8B/xCRMpzbX/sBkfX4ojEmYvyjoJhAUJl6ZSbs3QB718Ntz/gdlq/OeGUhIleKyAWqmg8MAV4A6oG3gJ0dEJ8xxnSoQFCZk1/MtZekMaB3EqzNhahYGPEpv0PzVWvNUL8Hat3pa4D/wBny4zAwy8O4jDHGF8u2HaC0osp5YjtQ77w69dKJkNTb79B81VozVLSqlrvT9wGzVHUuMFdE1ngbmjHGdLzcvCJ6J8Vx67C+UPg2nNgf0R3bDVq7sogWkYaEcjOwJGRZW/s7jDGmS9h/tJp/bt7PvTkZxMVEwZrZkNgLBk3wOzTftVbh5wLvishBnLuflgGIyCXAEY9jM8aYDvWPlSVux3YWVFXAx2/AFQ9ATJzfofnujMlCVX8gIm8DFwKL3IfowLkiedjr4IwxpqMEg0puXhGfuLg32WlJsPJ5CNTAqKl+h9YptNqUpKorminb6k04xhjjj/e3H6TkcBXfmeQ+S7EmF9IGQ7/Lz7xhhGjrQ3nGGBPWcvOK6JUUx4ThfaG8EIpXOIMGRtCrU8/EkoUxJuLtP1bN4k37uPeKDOJjomHtHEBg5BS/Q+s0LFkYYyLeSytLqG94YjsYdB7Eu2g8pDR9uWfksmRhjIlowaDyQn4xV1/Ui4vSu0PRcqgosmcrmrBkYYyJaMsLD7H7UKXzxDY4gwbGdYehd/gbWCdjycIYE9Fm5xXRs1ssE4dfALWVsPFVGDYZ4pL8Dq1TsWRhjIlYB4/XsGjjXu65PIOE2GjnIbzaY/ZsRTMsWRhjItbclSXUBZRpYzOdgrWzISULBozzN7BOyNNkISKTRGSLiGwXkcdaWGeKiGwSkY0iMjuk/AER2eZ+HvAyTmNM5FF1ntgeO7AXl/RJhqNlULgURt0HUfY7uinPBgMUkWic4cxvBUqAfBGZr6qbQtYZBDwOjFPVwyLSxy3vBXwXyMF52dJKd9vDXsVrjIksywsPsetQJd+8ZZBTsO5F0KDdBdUCL9PnWGC7qhaqai0wB7iryTpfAp5tSAKqut8tnwgsVtVyd9liYJKHsRpjIkxuXjEpibHcNuJCUHWerci8Cnpf7HdonZKXyaI/UBwyX+KWhboUuFREPhCRFSIy6Sy2NcaYc3LoeA0LN+zlnsv7Ox3be9bAgY+tY/sMvHwnRXMDqmiT+RhgEDAeyACWiciINm6LiEwHpgNkZWWdT6zGmAjy8qpSagPBU89WrMmF6HgYfo+/gXViXl5ZlACZIfMZQFkz67yqqnWquhPYgpM82rItqjpLVXNUNSc9Pb1dgzfGhKeGju2cAT25tG8y1NfChpdgyO2QmOp3eJ2Wl8kiHxgkItkiEgdMBeY3WWcecCOAiKThNEsVAguBCSLSU0R6AhPcMmOMOS8f7Syn8OCJU1cV2xdD5SHr2G6FZ81QqlovIg/hVPLRwHOqulFEngYKVHU+p5LCJiAAzFDVQwAi8t84CQfg6ZB3gRtjzDnLzSuiR0IM/3LZhU7BmtmQlA4X3+xvYJ2cp+/RVtUFwIImZU+GTCvwbffTdNvngOe8jM8YE1kOn6jlzfV7uf+qLKdju7Icti6EsdMh2tPqsMuzJ0+MMRFj7qoSagNBpjY8sb1hLgTrnJccmTOyZGGMiQiqypz8Yi7PSmXIBT2cwrW50HckXDDS3+C6AEsWxpiIULD7MNv3Hz/VsX1gK5SutGcr2siShTEmIuR+VERyQgx3XNbPKVg7GyQaRn7a38C6CEsWxpiwV1FZy+vr93D3mP4kxkVDMOCMBXXJzZDc1+/wugRLFsaYsPfK6lJq64NMvdJtgtr5HhwttWcrzoIlC2NMWGt4YntUZirD+jV0bM+B+BQYfLu/wXUhliyMMWFtVdFhtu47zv0Nt8vWHIPN82HE3RCb4G9wXYglC2NMWJv9UTHd40M6tjfNh7pKGHW/v4F1MZYsjDFh60hlHa+vK+Ou0f1Iinef0F6bC70ugsyx/gbXxViyMMaErXlrSqmpDxmKvKIIdi1zOraluTchmJZYsjDGhKWGju3LMlIY0T/FKVz7gvN92X3+BdZFWbIwxoSl1cUVfLz32KmrioZXpw64FnoO8De4LsiShTEmLOV+VERSXDSfHOV2bJfkQ/kOGzTwHFmyMMaEnaPVdby2row7R/ene2jHdkwiDL3T3+C6KEsWxpiw8+rqUqrrgtzf0ARVV+0MRz70k5DQw9/guihLFsaYsKKqzM4rZkT/HozMcDu2t74J1UesCeo8WLIwxoSVdSVH2Lzn6KmObXCG90juB9k3+BdYF+dpshCRSSKyRUS2i8hjzSz/vIgcEJE17uffQpYFQsrnexmnMabrm7e6lHEzl3DXsx8gQHTDYxTH98O2xXDZFIiK9jPELs2zl86KSDTwLHArUALki8h8Vd3UZNUXVPWhZnZRpaqjvYrPGBM+5q0u5fGX11NVFwBAge+9tpmE2BgmV78KGrARZs+Tl28oHwtsV9VCABGZA9wFNE0WxhhzRvWBIMdr6jlWXc/R6jqOVde7H2f6J4u2nEwUDarqAjyzcAuTU2dDvzHQZ4hP0YcHL5NFf6A4ZL4EuKqZ9T4lItcDW4FHVLVhmwQRKQDqgZmqOs/DWI0xHgkEleONKnn3u8b5PlrlfocuC0kIR6vrqKwNtH6gZvQ4sgWq18Ntz7TzWUUeL5NFcwOvaJP514BcVa0Rka8AfwFucpdlqWqZiFwELBGR9aq6o9EBRKYD0wGysrIwxjjmrS7lmYVbKKuool9qIjMmDmbymP5nvZ9gUDlW07jyDq3Qj7bwS7/h+2hfJh5bAAAWmUlEQVRVHSfaUNHHxUTRIyGG5ITYk999eySQ7E6HfvcIWccpj+GTv36fsiPVp+33c92Wg8bCiE+d9bmbxrxMFiVAZsh8BlAWuoKqHgqZ/QPwo5BlZe53oYgsBcYAO5psPwuYBZCTk9M0ERkTkZq235dWVPGduesoPHic0ZmpJ3/FN/yib+7XfEMiOF5T3+rx4qKjTlbaDRV4elr3JhW9U8knJ8TQIzH2tGXxMefX8fzvk4Y0OmeA7rFwT8z7kD0Rknqf1/6Nt8kiHxgkItlAKTAVaDSAvIhcqKp73Nk7gc1ueU+g0r3iSAPGAT/2MFYTxtrrV3Z7CQaV6voAVbUBquoCVNcFqKoNUlXnzFfVBqgJWV5VF6A6ZLqqNuhsU9d4Hw1l+4/WnHYJX1Mf5Fdvbz8tlthoaVRpJ8fHMqB3t0YVeo8miSC5yS/7hFj/7zBq+HuG/p2fGb2PhBWHrGO7nXiWLFS1XkQeAhYC0cBzqrpRRJ4GClR1PvANEbkTp1+iHPi8u/lQ4PciEsS5vXdmM3dRGdOq5n5lP/7yeoDTEoaqUlMfbFRJV9UGTquYa+qCZ1zeeD5ITV2gSSIInvV5iEBCTDSJcdEkxkaTEBt1cjo5IYY+yfEn5+fkF7e4n1e+9gmnsk90funHx0QhYTJU9+Qx/Rv/Tf/xICT2gkET/AsqjIhqeLTe5OTkaEFBgd9hGJ8Eg8qx6nrKK2spP1HL4RO1lFfW8v3XN3G0+vSmlNhoIbNXt1MVv1vRn4uE2CgSY91KPC66SaXeMB11cnnDuonuuo3LopxtYhvv42wq9XEzl1BaUXVaef/URD547KZmtghDVRXwk0vhigfgduvcPhMRWamqOa2t52UzlDHnRFWprA04lX5D5V9ZS/mJupNJ4PCJJuWVtQSCbf/hUxdQhl3Yo1HFfHLa/dXeXKWdEDKf6FbiUVGd65f5jImDT2u/T4yNZsbEwT5G1cE2zYNADYya6nckYcOShfFcTX2Awyfqmqn8G64AnCRwKOSKoLaFpproKKFnt1h6doujV1IcF6d3J2dgHL26xdEzKY5eSaeW9ewWx5TfL2dPM3fJ9E9N5H/vv9zrU/dFc+33fvfTdLg1uZA2GPqF59/YD5YsItD5dPjWB4JUVNU1+8u+ofI/1CQZnOnWydRusScr+v6piYzs38Op9Bsq/5NJwJlOTog5q1/y32nmLplI+JV9Wvt9JCkvhOIVcMtT9urUdmTJIsK0dFvlzoMnGNk/pVETT+MrAefK4EhVXYv77h4fQ88kp/LvlRTHJendT1b0zq/9kF/9SXGkJsYSE+3tWJb2KzsCrZ0DCIyc4nckYcWSRYT58Vsfn9aRW1Mf5Jdvb2tUFhcTRe+TlXwc/Xt2o1e32CaVv/Pdu3scqd1iz/teea9E9K/sSHOkDN7/BWRdAyn2N29PliwihKqycOO+Zp9ybfDaQ9c6VwZJcSTGRofNLZUmgiz4ttOxHR3rdyRhx5JFBFhXUsH3X99M3q5yYqKE+mbuGuqfmnjqRTHGdBW1lU7/xN/vdUaWbbDzXXgqBWLi4b/2+xdfGLFkEcZKK6p45q2PmbemjLTucfzw7pEkxAj/OW9jxHX4mjBRVw0l+bBrGexc5kwH64Ao5wG8mqMQrHfftX0HTPiB3xGHDUsWYeh4TT2/XbqdPy7bCcDXb7yYr46/5OSL66OioqzD13QN9bVQtgp2vud8SvKhvhokCi4cBVd/FbKvh6yrYdGTsOp5iElwmqLie0ByX7/PIGxYsggj9YEgLxQU8/PFWzl4vJa7x/Tn0YmD6Z+a2Gg96/A1nVagHvascRLDrmVQtALqKp1lfUdCzhdg4HUw4BOQmNp42xP74YoHIedBKPgzHN/X8fGHMUsWYWLplv38cMFmtu47ztiBvfjTA0MZlZna+obG+CkYgL3r3Wal92D3cqg95ixLHwpjPuskh4HXQrdeZ97X1P87NX3Hz7yLOUJZsujiPt57lB+8sZll2w4ysHc3fvfZK5g4vK/dyWQ6p2AQ9m861eew+32oPuIs630JjLwXsq9zEkT3Pv7GahqxZNFF7T9Wzc8Xb+WF/GKSE2J54o5hfO7qAcTFePuQmzFnRRUObj3V57D7A6h0X2PTcyAMvdPpcxh4LfTo52uo5swsWXQxVbUB/vR+Ib9duoPaQJAHx2Xz8E2XkNotzu/QjHGSQ3nhqT6HXe+f6jvokeEMFz7wOufqIdXebtmVWLLoIoJBZd4aZ0ynPUeqmTT8Ah67bQgD05L8Ds1EusO7T/U57FwGx9wXYna/wL1qcJNDz2wbq6kLs2TRBawoPMQP3tjM+tIjXJaRwi+njmFsdiudfcZ45UjpqT6HXe9BRZFT3i3NaU7Kvg6yb3D6ICw5hA1LFp3YzoMn+J8Fm1m0aR/9UhL4xX2juXNUv073/gQT5o7tc5uU3KuH8kKnPCHVSQ7XPORcPfQZaskhjFmy6IQqKmv55dvb+Nvy3cTHRDFj4mC+eG12p3jXsYkAJw6FJIdlcHCLUx7fw3m+4cp/c5JD3xEQZTdURApLFp1ITX2Avy3fza/e3sbxmnqmjs3ikVsuJT053u/QTDirOgy7P3QSw873YP9Gpzw2CQZcA6Pvd5qWLhgF0VZlRCpP//IiMgn4JRAN/FFVZzZZ/nngGaDULfpfVf2ju+wB4L/c8u+r6l+8jNVPqspbG/byP29+TFF5JTdcms5/3D6UwRck+x1a+Di2F156EO59PnKGgGjpnKuPQtHyU3cs7VkHqDNMRuZVcNMTTsd0vzE2eqs5ybNkISLRwLPArUAJkC8i81V1U5NVX1DVh5ps2wv4LpADKLDS3fawV/H6ZU1xBT94YxP5uw4zuG8yf/3CWK6/NN3vsMLP0pnO08FL/hsmzXTa1iUKcL8l6lRZuLS7v/tjZ7iMd74PwyafalYqW+2M0BodBxljYfxjTrNSRo4zSqsxzfDyymIssF1VCwFEZA5wF9A0WTRnIrBYVcvdbRcDk4Bcj2LtcCWHK3lm4RZeXVNGWvd4/ueekUzJySTaOq9PCdRBzTGoPe5817jftaHTx52RRhvNHzs13dAZ22D135xPqyQkeYQmlaZJRs6ceE6W0YZ9hax3xn2FJrVm9rXjbdCQd5iv+qvzAefK4bpvO8khcyzENh43zJiWeJks+gPFIfMlwFXNrPcpEbke2Ao8oqrFLWwbFiPfHauu4zdLd/Cn93ciwMM3XcKXb7j45IiwHReIR80yDRX8aZX8UXc+tMJvodJv2K6+5Rc1NRKTCPHdIT4Z4ro7HbE9+jnTGVc6v6TLC52hq6Ninbt2Bk1wttGg8yCZKqDufENZMKQsZBk0U6Zn2O5M+2rDeqfF5c4HW9hX+lA4WuoOo6EQFeO8Oe6Tv4LeF7Xf39pEFC9rqOZ+Ijd9685rQK6q1ojIV4C/ADe1cVtEZDowHSArq3M/DVofCDIn3xkR9tCJWu5xR4Ttl+rTL7uGJop3f+Q0yzT9Rd4wfXI+tMJv+iv/2KlK/qwq+OSQSj4ZevQPqfCTT31Oznd31mu6XWudrq89Aoe2u0NX1zoJ5OYnzv/fsDN77RFnuO7oeOec0y61RGHOi5fJogTIDJnPAMpCV1DVQyGzfwB+FLLt+CbbLm16AFWdBcwCyMnJOf31b52AqrJ0ywF+uGAz2/YfZ2x2L/78L0O5LKMDRoQN1Du/MI8UOw9OVRQ7ySH0jWIFf3I+bRHb7fSKu0fG6b/qG823UOF35F01kTh0dSSes/GUqHpTx4pIDE7T0s04dzvlA/er6saQdS5U1T3u9N3Ad1T1areDeyVwubvqKuCKhj6M5uTk5GhBQYEn53KuNu85yg8XOCPCZqcl8dhtQ5gwrB1HhK2vdZJBRZHzOZkU3MRwtLRxYgBI6uM0x1QfcZZFxUDaEBg+GXpc2HIlH9fdbps0JgyJyEpVzWltPc/+71fVehF5CFiIc+vsc6q6UUSeBgpUdT7wDRG5E6gHyoHPu9uWi8h/4yQYgKfPlCg6m/3HqvnZoq28WOCMCPvkHcP47LmMCFtfA0dKoGK3U/k3TQpHy2jcOidOU05qpnN/fGoWpGQ636lZkJLh3O3S0ETR0CyTdRXcMKMd/wWMMeHGsyuLjtYZriyqagP8YVkhv3t3B3WBIA9cM5CHbxpESrcW7lWvqzqVBI4UNb4qqCiC43sbry/RbjJwK//U0ESQ6SyLacPos3M+A937Nm6iCH1xjDEmYrT1ysKSRTsIBpVXVjsjwu49Ws1tI5wRYQd0V/cqoNi9OmjSVHTiQOMdRcVCSkgySMlqnBiS+1lTkDGmXfneDNWlnOttpNVHWbthHa++u4JAeRGPJR/l+sFV9DqxF/5YBFVNWs6i4041Cw2+zU0CA06VJV8AUTb+kzGm87FkAY1vI214d68qVFc031dQsZvA4WKiayoYBYwCiAWtT0BOuFcC/caEXBW4n6Q+NvCaMaZLiuxk8f0+Tidyg5O3kYpzB1DN0cbrxyYRSMlkR21P8iqvZF9UX0YMG874q3KIT8tGktLCZ6gIY4wJEdnJ4pvrYMEM2Dz/VFlCCvS7AtIGNepErumewV9XH+XX72zneE0908Zm8citl5LW3cbSMcaEv8hOFskXQLfegDj9CcE6GHHvqaYonIfqFqzfy4/+voGi8krGD3ZGhL20r40Ia4yJHJGdLMC5IynnC80+6bq66DDff2MzK3cfZsgFyfzti2O5bpCNCGuMiTyWLEKfL3CvKIrLK/nxwi28traM9OR4Zt4zkk/biLDGmAgW8clinvt8RFlFFRekJDDswh4s236QKIFvuCPCJnX0iLDGGNPJRHQtOG91KY+/vJ6qOmf8pD1HqtlzpJqcAan8+v7LuTDFxvo3xhiAiL7p/5mFW04milB7jtRYojDGmBARnSzKKqrOqtwYYyJVRCeLll485NsLiYwxppOK6GQxY+JgEmMbj8WUGBvNjImDfYrIGGM6p4ju4J48xnmtd8PdUP1SE5kxcfDJcmOMMY6IThbgJAxLDsYYc2YR3QxljDGmbSxZGGOMaZUlC2OMMa2yZGGMMaZVliyMMca0SlTV7xjahYgcAHafxy7SgIPtFE5XEWnnHGnnC3bOkeJ8znmAqrb67oWwSRbnS0QKVDXH7zg6UqSdc6SdL9g5R4qOOGdrhjLGGNMqSxbGGGNaZcnilFl+B+CDSDvnSDtfsHOOFJ6fs/VZGGOMaZVdWRhjjGlVxCcLEXlORPaLyAa/Y+kIIpIpIu+IyGYR2Sgi3/Q7Jq+JSIKI5InIWvecv+d3TB1FRKJFZLWIvO53LB1BRHaJyHoRWSMiBX7H0xFEJFVEXhKRj93/r6/x5DiR3gwlItcDx4G/quoIv+PxmohcCFyoqqtEJBlYCUxW1U0+h+YZEREgSVWPi0gs8D7wTVVd4XNonhORbwM5QA9VvcPveLwmIruAHFWNmOcsROQvwDJV/aOIxAHdVLWivY8T8VcWqvoeUO53HB1FVfeo6ip3+hiwGQjrMdrVcdydjXU/Yf8rSUQygH8B/uh3LMYbItIDuB74E4Cq1nqRKMCSRUQTkYHAGOAjfyPxntscswbYDyxW1bA/Z+AXwL8DQb8D6UAKLBKRlSIy3e9gOsBFwAHgz25z4x9FJMmLA1myiFAi0h2YC3xLVY/6HY/XVDWgqqOBDGCsiIR1k6OI3AHsV9WVfsfSwcap6uXAbcDX3WbmcBYDXA78VlXHACeAx7w4kCWLCOS2288F/k9VX/Y7no7kXqIvBSb5HIrXxgF3um34c4CbROTv/obkPVUtc7/3A68AY/2NyHMlQEnIlfJLOMmj3VmyiDBuZ++fgM2q+jO/4+kIIpIuIqnudCJwC/Cxv1F5S1UfV9UMVR0ITAWWqOpnfQ7LUyKS5N60gdsUMwEI67scVXUvUCwig92imwFPblaJ+Hdwi0guMB5IE5ES4Luq+id/o/LUOOBzwHq3DR/gP1R1gY8xee1C4C8iEo3zA+lFVY2IW0kjTF/gFef3EDHAbFV9y9+QOsTDwP+5d0IVAg96cZCIv3XWGGNM66wZyhhjTKssWRhjjGmVJQtjjDGtsmRhjDGmVZYsjDHGtMqShelQIqIi8tOQ+UdF5Kl22vfzInJve+yrleN82h3d851mlj3jjmz7zDnsd7SI3N4+UfpHRAZGyijOkcSSheloNcA9IpLmdyCh3Gcw2uqLwNdU9cZmln0ZuFxVZ5xDGKOBs0oW4rD/j43n7D8y09HqcV4B+UjTBU2vDETkuPs9XkTeFZEXRWSriMwUkc+476hYLyIXh+zmFhFZ5q53h7t9tPuLP19E1onIl0P2+46IzAbWNxPPNHf/G0TkR27Zk8C1wO+aXj2IyHwgCfhIRO5znxyf6x43X0TGueuNFZEP3YHfPhSRwe4DVU8D97nvYrhPRJ4SkUdD9r/B/dU+0L2y+Q2wCsgUkQkislxEVonIP9yxv3D/rTa55/2TZs6xpWMkicgb4rwDZIOI3Ocuv8L9W6wUkYXiDHnfUL5WRJYDX2/hb2+6MlW1j3067IPz7pAewC4gBXgUeMpd9jxwb+i67vd4oALnSex4oBT4nrvsm8AvQrZ/C+dH0CCccXMSgOnAf7nrxAMFQLa73xNAdjNx9gOKgHScp4GX4Lz3A5yxpXJaOr+Q6dnAte50Fs4QK7jnH+NO3wLMdac/D/xvyPZPAY+GzG8ABrqfIHC1W54GvIfzzg6A7wBPAr2ALZx6+Da1mXhbOsangD+ElKfgDO3+IZDult0HPOdOrwNucKefATb4/d+afdr3E/HDfZiOp6pHReSvwDeAqjZulq+qewBEZAewyC1fD4Q2B72oqkFgm4gUAkNwxgi6LOSqJQUnmdQCeaq6s5njXQksVdUD7jH/D+e9AfPaGC84iWCYO/wEQA937KIUnOFHBuEMqR17FvtssFtPvbzpamAY8IF7rDhgOXAUqAb+KCJvAGczxMl64CfuFdXrqrpMnJF6RwCL3eNEA3tEJAUnEb3rbvs3nFFfTRixZGH88gucJpQ/h5TV4zaNilMbxYUsqwmZDobMB2n833HT8WsUEOBhVV0YukBExuNcWTRHWig/G1HANaraKCGKyK+Bd1T1bnHeKbK0he1P/nu4EkKmQ+MWnHd0TGu6AxEZizO43FTgIeCmthxDVbeKyBU4fSj/IyKLcEZx3aiqjV7bKc4gjTZuUJizPgvjC1UtB17E6SxusAu4wp2+i3P7xf1pEYly+zEuwmmGWQh8VZyh2RGRS6X1F8R8BNwgImlu5/c04N1WtmlqEU4FjXvc0e5kCk5TGjhNTw2OAckh87twh5sWkctxms6aswIYJyKXuOt2c8+xO5CiziCR38LpQG+q2WOISD+gUlX/DvzEXWcLkC7uO55FJFZEhqsz7PsREbnW3ednWojTdGGWLIyfforT3t7gDzgVdB5wFS3/6j+TLTiV+pvAV1S1Gue1opuAVeLc0vl7Wrmqdpu8HgfeAdYCq1T11bOM5RtAjtu5vAn4ilv+Y5xf6x/gNOU0eAen2WqN26E8F+glzujAXwW2thDrAZykkysi63CSxxCcxPO6W/YuzdxUcIZjjATy3PL/BL6vqrXAvcCPRGQtsAb4hLv+g8Czbgd3W5sWTRdio84aY4xplV1ZGGOMaZUlC2OMMa2yZGGMMaZVliyMMca0ypKFMcaYVlmyMMYY0ypLFsYYY1plycIYY0yr/j+jO/FDj3VOCwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(features_num_seq, result_training_scores, marker='o', label='train')\n",
    "plt.plot(features_num_seq, result_test_scores, marker='*', label='test')\n",
    "\n",
    "plt.xticks(features_num_seq)\n",
    "plt.legend()\n",
    "plt.xlabel('Number of features used')\n",
    "plt.ylabel('Score')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20190402</td>\n",
       "      <td>20</td>\n",
       "      <td>21</td>\n",
       "      <td>25</td>\n",
       "      <td>24</td>\n",
       "      <td>25</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20190403</td>\n",
       "      <td>21</td>\n",
       "      <td>22</td>\n",
       "      <td>25</td>\n",
       "      <td>26</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>20190404</td>\n",
       "      <td>23</td>\n",
       "      <td>23</td>\n",
       "      <td>24</td>\n",
       "      <td>26</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>20190405</td>\n",
       "      <td>23</td>\n",
       "      <td>23</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20190406</td>\n",
       "      <td>25</td>\n",
       "      <td>25</td>\n",
       "      <td>28</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          0   1   2   3   4   5  6\n",
       "0  20190402  20  21  25  24  25  0\n",
       "1  20190403  21  22  25  26  27  0\n",
       "2  20190404  23  23  24  26  27  1\n",
       "3  20190405  23  23  27  27  27  1\n",
       "4  20190406  25  25  28  27  27  1"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test=pd.read_csv(\"prediction.csv\", delim_whitespace=True, header=None)\n",
    "df_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    20190402\n",
       "1    20190403\n",
       "2    20190404\n",
       "3    20190405\n",
       "4    20190406\n",
       "5    20190407\n",
       "6    20190408\n",
       "Name: Date, dtype: int64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test.columns=features\n",
    "df_test.head()\n",
    "Date=df_test[\"Date\"]\n",
    "Date\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>TaipeiTemp</th>\n",
       "      <th>TaoyuanTemp</th>\n",
       "      <th>TaichungTemp</th>\n",
       "      <th>TainanTemp</th>\n",
       "      <th>KaohsingTemp</th>\n",
       "      <th>vacation</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>21</td>\n",
       "      <td>25</td>\n",
       "      <td>24</td>\n",
       "      <td>25</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>21</td>\n",
       "      <td>22</td>\n",
       "      <td>25</td>\n",
       "      <td>26</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>23</td>\n",
       "      <td>23</td>\n",
       "      <td>24</td>\n",
       "      <td>26</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>23</td>\n",
       "      <td>23</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>25</td>\n",
       "      <td>25</td>\n",
       "      <td>28</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   TaipeiTemp  TaoyuanTemp  TaichungTemp  TainanTemp  KaohsingTemp  vacation\n",
       "0          20           21            25          24            25         0\n",
       "1          21           22            25          26            27         0\n",
       "2          23           23            24          26            27         1\n",
       "3          23           23            27          27            27         1\n",
       "4          25           25            28          27            27         1"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test.drop(columns=[\"Date\"],inplace=True)\n",
    "df_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(7,)\n",
      "(7, 6)\n"
     ]
    }
   ],
   "source": [
    "x=df_test.values\n",
    "y_te_pred=reg.predict(x)\n",
    "print(y_te_pred.shape)\n",
    "print(x.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>TaipeiTemp</th>\n",
       "      <th>TaoyuanTemp</th>\n",
       "      <th>TaichungTemp</th>\n",
       "      <th>TainanTemp</th>\n",
       "      <th>KaohsingTemp</th>\n",
       "      <th>vacation</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>21</td>\n",
       "      <td>25</td>\n",
       "      <td>24</td>\n",
       "      <td>25</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>21</td>\n",
       "      <td>22</td>\n",
       "      <td>25</td>\n",
       "      <td>26</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>23</td>\n",
       "      <td>23</td>\n",
       "      <td>24</td>\n",
       "      <td>26</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>23</td>\n",
       "      <td>23</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>25</td>\n",
       "      <td>25</td>\n",
       "      <td>28</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   TaipeiTemp  TaoyuanTemp  TaichungTemp  TainanTemp  KaohsingTemp  vacation\n",
       "0          20           21            25          24            25         0\n",
       "1          21           22            25          26            27         0\n",
       "2          23           23            24          26            27         1\n",
       "3          23           23            27          27            27         1\n",
       "4          25           25            28          27            27         1\n",
       "5          24           24            27          27            27         1\n",
       "6          24           24            27          27            27         0"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>peak(HW)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>29120.377125</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>30006.424873</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>26322.046475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>25660.612372</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>26409.734375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>26122.430857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>30415.483159</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       peak(HW)\n",
       "0  29120.377125\n",
       "1  30006.424873\n",
       "2  26322.046475\n",
       "3  25660.612372\n",
       "4  26409.734375\n",
       "5  26122.430857\n",
       "6  30415.483159"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prediction = pd.DataFrame(y_te_pred, columns=[\"peak(HW)\"])\n",
    "prediction\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Date', 'peak(HW)'], dtype='object')"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = pd.concat([ Date, prediction], axis=1)\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "result.to_csv(\"submission.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
