datapoints = [
  [[1.2, 0.7], 1],
  [[-0.3, -0.5], -1],
  [[3.0, 0.1], 1],
  [[-0.1, -1.0], -1],
  [[-1.0, 1.1], -1],
  [[2.1, -3.0], 1]
];

class NN{
  constructor(){
    this.a11 = (Math.random() - 0.5) * 2;
    this.a12 = (Math.random() - 0.5) * 2;
    this.a13 = (Math.random() - 0.5) * 2;

    this.b11 = (Math.random() - 0.5) * 2;
    this.b12 = (Math.random() - 0.5) * 2;
    this.b13 = (Math.random() - 0.5) * 2;

    this.c11 = (Math.random() - 0.5) * 2;
    this.c12 = (Math.random() - 0.5) * 2;
    this.c13 = (Math.random() - 0.5) * 2;

    this.a2 = (Math.random() - 0.5) * 2;
    this.b2 = (Math.random() - 0.5) * 2;
    this.c2 = (Math.random() - 0.5) * 2;
    this.d2 = (Math.random() - 0.5) * 2;

    this.step_size = 0.01;
  }

  forward(x, y){
    let h1 = Math.max(0, x * this.a11 + y * this.b11 + this.c11);
    let h2 = Math.max(0, x * this.a12 + y * this.b12 + this.c12);
    let h3 = Math.max(0, x * this.a13 + y * this.b13 + this.c13);
    return h1 * this.a2 + h2 * this.b2 + h3 * this.c2 + this.d2;
  }

  train(x, y, l){
    let h1 = Math.max(0, x * this.a11 + y * this.b11 + this.c11);
    let h2 = Math.max(0, x * this.a12 + y * this.b12 + this.c12);
    let h3 = Math.max(0, x * this.a13 + y * this.b13 + this.c13);
    let o = h1 * this.a2 + h2 * this.b2 + h3 * this.c2 + this.d2;
    let pull = l - o;

    let da2 = pull * h1 - this.a2;
    let db2 = pull * h2 - this.b2;
    let dc2 = pull * h3 - this.c2;
    let dd2 = pull * 1.0;

    let da11 = (h1 > 0 ? da2 * x : 0) - this.a11;
    let da12 = (h2 > 0 ? db2 * x : 0) - this.a12;
    let da13 = (h3 > 0 ? dc2 * x : 0) - this.a13;
    let db11 = (h1 > 0 ? da2 * y : 0) - this.b11;
    let db12 = (h2 > 0 ? db2 * y : 0) - this.b12;
    let db13 = (h3 > 0 ? dc2 * y : 0) - this.b13;
    let dc11 = h1 > 0 ? 1 : 0;
    let dc13 = h2 > 0 ? 1 : 0;
    let dc12 = h3 > 0 ? 1 : 0;

    this.a11 += da11 * this.step_size;
    this.a12 += da12 * this.step_size;
    this.a13 += da13 * this.step_size;
    this.b11 += db11 * this.step_size;
    this.b12 += db12 * this.step_size;
    this.b13 += db13 * this.step_size;
    this.c11 += dc11 * this.step_size;
    this.c12 += dc12 * this.step_size;
    this.c13 += dc13 * this.step_size;
    this.a2 += da2 * this.step_size;
    this.b2 += db2 * this.step_size;
    this.c2 += dc2 * this.step_size;
    this.d2 += dd2 * this.step_size;
  }

  accuracy(datapoints){
    let N = datapoints.length;
    let n = 0;
    for(let i = 0; i < N; i++){
      let x = datapoints[i][0][0];
      let y = datapoints[i][0][1];
      let l = datapoints[i][1];

      let o = this.forward(x, y);
      if(o > 0 ? 1 : -1 === l) n += 1;
    }
    return n / N;
  }
}

let nn = new NN();
let iterations = 401;
for(let i = 0; i < iterations; i++){
  let train_p = datapoints[Math.floor(Math.random() * datapoints.length)];
  let x = train_p[0][0];
  let y = train_p[0][1];
  let l = train_p[1];
  nn.train(x, y, l);
  if(i % 100 === 0){
    console.log(i, nn.accuracy(datapoints) * 100);
  }
}
