import requests
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import update
app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///C:\\Users\\Risha\\OneDrive\\Desktop\\ALL\\minor\\sid\\mainwebsite\\Info.db"
app.config['SQLALCHEMY_BINDS']={'db2' : "sqlite:///C:\\Users\\Risha\\OneDrive\\Desktop\\ALL\\minor\\sid\\mainwebsite\\Discussion.db"}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)
class Info(db.Model):
    Sno=db.Column(db.Integer,primary_key=True)
    Crop_name=db.Column(db.String(100),nullable=False)
    Info_type=db.Column(db.String(100),nullable=False)
    Ans=db.Column(db.String(1000),nullable=True)
    def __repr__(self) -> str:
        return f"{self.Ans}"
class Discussions(db.Model):
    __bind_key__ ='db2'
    Sno=db.Column(db.Integer,primary_key=True)
    Farmer_name=db.Column(db.String(100),nullable=False)
    Query=db.Column(db.String(1000),nullable=False)
    Ans=db.Column(db.String(1000),nullable=True)
    def __repr__(self) -> str:
        return f"{self.Ans}"

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/getinfo')
def getinfo():
    return render_template("getinfo.html")
@app.route('/about')
def about():
    return render_template("about.html")
@app.route('/predict')
def predict():
   return render_template("predict.html")
@app.route('/answer2',methods=['POST','GET'])
def answer2():
    if request.method=='POST':
        location=request.form['location']
        result=requests.get('http://api.openweathermap.org/geo/1.0/direct?q={0}&appid=f077f7a61c54b4c76e0c07c2989ba7f6'.format(location))
        data=result.json()
        dict=data[0]
        lat=dict['lat']
        lon=dict['lon']
        result1=requests.get('https://api.openweathermap.org/data/2.5/weather?lat={0}&lon={1}&appid=f077f7a61c54b4c76e0c07c2989ba7f6'.format(lat,lon))
        data1=result1.json()
        description=data1['weather'][0]['description']
        temperature=data1['main']['temp']
        humidity=data1['main']['humidity']
        visibility=data1['visibility']
        clouds=data1['clouds']['all']
        windspeed=data1['wind']['speed']
        pressure=data1['main']['pressure']
        loc_=data1['name']
        return render_template('answer2.html',description=description,temperature=(temperature-273.15),humidity=humidity,visibility=visibility,clouds=clouds,windspeed=windspeed,pressure=pressure,loc_=loc_)
    return render_template('answer2.html')
@app.route('/answer1',methods=['POST','GET'])
def answer1():
    if request.method=='POST':
        a=request.form.get('crop_name')
        b=request.form.get('info_type')
        if (a=='23' or b=='9'):
            return render_template("answer1.html",c='Invalid Input')
        arr_crop=['Rice','Maize','Chickpea','Kidneybeans','Pigeonpeas','Mothbeans','Mungbean','Blackgram','Lentil','Pomegranate','Banana','Mango','Grapes','Watermelon','Muskmelon','Apple','Orange','Papaya','Coconut','Cotton','Jute','Coffee']
        arr_info=['Crop Plan','Seeds Information','Soil Needs','Irrigation Strategy','Protecting the crop','Machinery Required','Storage Information','MSP of crop']
        a_new=arr_crop[int(a)-1]
        b_new=arr_info[int(b)-1]
        c=Info.query.filter(Info.Crop_name==a_new).filter(Info.Info_type==b_new).all()
        return render_template("answer1.html",c=c,a_new=a_new,b_new=b_new)
    return render_template('answer1.html')

@app.route('/weather')
def weather():
    return render_template('weather.html')
@app.route('/disease')
def disease():
    return render_template('disease.html')
@app.route('/answer3', methods=['GET','POST'])
def answer3():
    if request.method=='POST':
        image=request.form['imag']
        import base64
        with open(image, "rb") as file:
            images = [base64.b64encode(file.read()).decode("ascii")]

        response = requests.post(
            "https://api.plant.id/v2/health_assessment",
            json={
                "images": images,
                "modifiers": ["similar_images"],
                "disease_details": ["description", "treatment"],
            },
            headers={
                "Content-Type": "application/json",
                "Api-Key": "3TlT0niIlvfeK4sHmWtzCM80e7aaUOhsoe83xjDgKVlDAF4Koq",
            }).json()

        if not response["health_assessment"]["is_healthy"]:
            for suggestion in response["health_assessment"]["diseases"]:
                probability=suggestion["probability"]
                name=suggestion["name"]
                description=suggestion["disease_details"]["description"]
                treat_dict=suggestion["disease_details"]["treatment"]
                prevention=treat_dict["prevention"]
                return render_template('answer3.html',probability=probability,name=name,description=description,prevention=prevention)
            return render_template('answer3.html')
@app.route('/discussions',methods=['POST','GET'])
def discussions():
    if (request.method=='POST'):
        name=request.form['farmername']
        query=request.form['query']
        discu=Discussions(Farmer_name=name,Query=query)
        db.session.add(discu)
        db.session.commit()
    alldis=Discussions.query.all()
    return render_template('discussions.html',alldis=alldis)

@app.route('/viewans/<Sno>',methods=['POST','GET'])
def viewans(Sno):
    answer=Discussions.query.filter_by(Sno=Sno).first()
    answer1=str(answer)#remember to make string as it was coming as object
    x=answer1.split(',')
    return render_template ('viewans.html',x=x)
@app.route('/ans/<Sno>',methods=['POST','GET'])
def ans(Sno):
    return render_template('answer.html',Sno=Sno)
@app.route('/answerques/<Sno>',methods=['POST','GET'])
def addans(Sno):
    if (request.method=='POST'):
        discu=Discussions.query.filter_by(Sno=Sno).first()
        ans1=discu.Ans
        if (ans1==None):
            ans1=" "
        ans2=request.form['ans']
        name=request.form['name']
        discu.Ans=ans1+','+ans2+"-answer by "+name
        db.session.add(discu)
        db.session.commit()
    return render_template('answer.html')
@app.route('/predict1',methods=['POST','GET'])
def predict1():
    if(request.method=='POST'):
        from collections import Counter
        import numpy as np


        def euclidean_distance(x1, x2):
            return np.sqrt(np.sum((x1 - x2) ** 2))


        class KNN:
            def __init__(self, k=3):
                self.k = k

            def fit(self, X, y):
                self.X_train = X
                self.y_train = y

            def predict(self, X):
                y_pred = [self._predict(x) for x in X]
                return np.array(y_pred)

            def _predict(self, x):
                # Compute distances between x and all examples in the training set
                distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
                # Sort by distance and return indices of the first k neighbors
                k_idx = np.argsort(distances)[: self.k]
                # Extract the labels of the k nearest neighbor training samples
                k_neighbor_labels = [self.y_train[i] for i in k_idx]
                # return the most common class label
                most_common = Counter(k_neighbor_labels).most_common(1)
                return most_common[0][0]





        class NaiveBayes:
            def fit(self, X, y):
                n_samples, n_features = X.shape
                self._classes = np.unique(y)
                n_classes = len(self._classes)

                # calculate mean, var, and prior for each class
                self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
                self._var = np.zeros((n_classes, n_features), dtype=np.float64)
                self._priors = np.zeros(n_classes, dtype=np.float64)

                for idx, c in enumerate(self._classes):
                    X_c = X[y == c]
                    self._mean[idx, :] = X_c.mean(axis=0)
                    self._var[idx, :] = X_c.var(axis=0)
                    self._priors[idx] = X_c.shape[0] / float(n_samples)

            def predict(self, X):
                y_pred = [self._predict(x) for x in X]
                return np.array(y_pred)

            def _predict(self, x):
                posteriors = []

                # calculate posterior probability for each class
                for idx, c in enumerate(self._classes):
                    prior = np.log(self._priors[idx])
                    posterior = np.sum(np.log(self._pdf(idx, x)))
                    posterior = prior + posterior
                    posteriors.append(posterior)

                # return class with highest posterior probability
                return self._classes[np.argmax(posteriors)]

            def _pdf(self, class_idx, x):
                mean = self._mean[class_idx]
                var = self._var[class_idx]
                numerator = np.exp(-((x - mean) ** 2) / (2 * var))
                denominator = np.sqrt(2 * np.pi * var)
                return numerator / denominator





        def entropy(y):
            hist = np.bincount(y)
            ps = hist / len(y)
            return -np.sum([p * np.log2(p) for p in ps if p > 0])


        class Node:
            def __init__(
                self, feature=None, threshold=None, left=None, right=None, *, value=None
            ):
                self.feature = feature
                self.threshold = threshold
                self.left = left
                self.right = right
                self.value = value

            def is_leaf_node(self):
                return self.value is not None


        class DecisionTree:
            def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
                self.min_samples_split = min_samples_split
                self.max_depth = max_depth
                self.n_feats = n_feats
                self.root = None

            def fit(self, X, y):
                self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
                self.root = self._grow_tree(X, y)

            def predict(self, X):
                return np.array([self._traverse_tree(x, self.root) for x in X])

            def _grow_tree(self, X, y, depth=0):
                n_samples, n_features = X.shape
                n_labels = len(np.unique(y))

                # stopping criteria
                if (
                    depth >= self.max_depth
                    or n_labels == 1
                    or n_samples < self.min_samples_split
                ):
                    leaf_value = self._most_common_label(y)
                    return Node(value=leaf_value)

                feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

                # greedily select the best split according to information gain
                best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

                # grow the children that result from the split
                left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
                left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                return Node(best_feat, best_thresh, left, right)

            def _best_criteria(self, X, y, feat_idxs):
                best_gain = -1
                split_idx, split_thresh = None, None
                for feat_idx in feat_idxs:
                    X_column = X[:, feat_idx]
                    thresholds = np.unique(X_column)
                    for threshold in thresholds:
                        gain = self._information_gain(y, X_column, threshold)

                        if gain > best_gain:
                            best_gain = gain
                            split_idx = feat_idx
                            split_thresh = threshold

                return split_idx, split_thresh

            def _information_gain(self, y, X_column, split_thresh):
                # parent loss
                parent_entropy = entropy(y)

                # generate split
                left_idxs, right_idxs = self._split(X_column, split_thresh)

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    return 0

                # compute the weighted avg. of the loss for the children
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
                child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

                # information gain is difference in loss before vs. after split
                ig = parent_entropy - child_entropy
                return ig

            def _split(self, X_column, split_thresh):
                left_idxs = np.argwhere(X_column <= split_thresh).flatten()
                right_idxs = np.argwhere(X_column > split_thresh).flatten()
                return left_idxs, right_idxs

            def _traverse_tree(self, x, node):
                if node.is_leaf_node():
                    return node.value

                if x[node.feature] <= node.threshold:
                    return self._traverse_tree(x, node.left)
                return self._traverse_tree(x, node.right)

            def _most_common_label(self, y):
                counter = Counter(y)
                most_common = counter.most_common(1)[0][0]
                return most_common



        def accuracy(y_true, y_pred):
                accuracy = np.sum(y_true == y_pred) / len(y_true)
                return accuracy

        import pandas as pd
        dataset = pd.read_csv('Crop_recommendation_new.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)


        dt = DecisionTree(max_depth=10)
        dt.fit(X_train, y_train)
        y_predictions1 = dt.predict(X_test)
        acc = accuracy(y_test, y_predictions1)
        #print("Decision tree classification accuracy: ", acc)


        nb = NaiveBayes()
        nb.fit(X_train, y_train)
        y_predictions3 = nb.predict(X_test)
        #print("Naive Bayes classification accuracy", accuracy(y_test, y_predictions3))


        k = 3
        kn = KNN(k=k)
        kn.fit(X_train, y_train)
        y_predictions2 = kn.predict(X_test)
        #print("KNN classification accuracy", accuracy(y_test, y_predictions2))


        N=int(request.form.get('nitrogen'))
        P=int(request.form.get('phosphorous'))
        K=int(request.form.get('potassium'))
        temp=float(request.form.get('temperature'))
        H=float(request.form.get('humidity'))
        ph=float(request.form.get('ph'))
        rain=float(request.form.get('rainfall'))
        X_new=np.array([[N],[P],[K],[temp],[H],[ph],[rain]]).reshape(1,7)

        X_new=sc.transform(X_new)


        y_pred1=dt.predict(X_new)
        y_pred2=nb.predict(X_new)
        y_pred3=kn.predict(X_new)


        #print(y_pred1," ",y_pred2," ",y_pred3)

        predict_cmn=[y_pred1[0],y_pred2[0],y_pred3[0]]

        import statistics
        from statistics import mode

        vector_nam=["rice","maize","chickpea","kidneybeans","pigeonpeas","mothbeans","mungbeans","blackgram","lentil","pomengranate","banana",
        "mango","grapes","watermelon","muskmelon","apple","orange","papaya","coconut","cotton","jute","coffee"]

        predicted_crop=vector_nam[mode(predict_cmn)]
    return render_template('myprediction.html',predicted_crop=predicted_crop)

if __name__=="__main__":
    app.run(host='127.0.0.1',port=5001)