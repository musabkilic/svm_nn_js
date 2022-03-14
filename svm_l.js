datapoints = [
  [[1.2, 0.7], 1],
  [[-0.3, -0.5], -1],
  [[3.0, 0.1], 1],
  [[-0.1, -1.0], -1],
  [[-1.0, 1.1], -1],
  [[2.1, -3.0], 1]
];

class SVM{
  constructor(){
    this.a = (Math.random() - 0.5) * 2;
    this.b = (Math.random() - 0.5) * 2;
    this.c = (Math.random() - 0.5) * 2;

    this.alpha = 0.1;
    this.step_size = 0.01;
  }

  forward(x, y){
    return x * this.a + y * this.b + this.c;
  }

  train(x, y, l){
    let o = this.forward(x, y);
    let loss = Math.max(0, -l * o + 1);

    let pull = loss;

    let da = x * pull - this.alpha * this.a;
    let db = y * pull - this.alpha * this.b;
    let dc = 1 * pull;

    this.a += da * this.step_size;
    this.b += db * this.step_size;
    this.c += dc * this.step_size;
  }

  accuracy(datapoints){
    let N = datapoints.length;
    let n = 0;
    let total_loss = 0;
    for(let i = 0; i < N; i++){
      let x = datapoints[i][0][0];
      let y = datapoints[i][0][1];
      let l = datapoints[i][1];

      let o = this.forward(x, y);
      if(o > 0 ? 1 : -1 === l) n += 1;
      total_loss += Math.max(0, -l * o + 1);
    }
    total_loss += this.alpha * (this.a * this.a + this.b * this.b);

    return [n / N, total_loss];
  }
}

let svm = new SVM();
let iterations = 400;
for(let i = 0; i < iterations; i++){
  let train_p = datapoints[Math.floor(Math.random() * datapoints.length)];
  let x = train_p[0][0];
  let y = train_p[0][1];
  let l = train_p[1];
  svm.train(x, y, l);
  if(i % 100 === 0){
    console.log(i, ...svm.accuracy(datapoints));
  }
}
