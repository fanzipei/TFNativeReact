import { StatusBar } from 'expo-status-bar';
import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { PlatformReactNative } from '@tensorflow/tfjs-react-native';

function createModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [1],
      units: 1,
      useBias: true,
    })
  );
  return model;
}

function createData() {
  const x = tf.randomUniform([2000, 1]);
  const y = tf.mul(x, tf.scalar(2));
  return {
    x: tf.add(tf.randomNormal([2000, 1], 0, 0.01), x),
    y: y,
  };
}

async function trainModel(model, inputs, labels) {
  const batch_size = 32;
  const epochs = 10;
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });
  return await model.fit(inputs, labels, {
    batch_size,
    epochs,
    shuffle: true,
  });
}

async function predict(model, inputs) {
  return await model.predict(inputs);
}

export default function App() {
  const [trained, setTrained] = useState(false);
  const [model, setModel] = useState(null);
  const [data, setData] = useState(null);
  const [prediction, setPrediction] = useState(null);

  tf.setBackend('cpu');
  return (
    <View style={styles.container}>
      <Button
        title='Create Model and Data'
        onPress={() => {
          setModel(createModel());
          setData(createData());
          setTrained(false);
          setPrediction(null);
        }}
      />
      <Button
        title='Train'
        onPress={() => {
          async function training() {
            await trainModel(model, data.x, data.y);
            setTrained(true);
          }
          training();
        }}
      />
      <Button
        title='Predict'
        onPress={() => {
          async function predicting() {
            setPrediction(await predict(model, data.x));
          }
          predicting();
        }}
      />
      <Text>inputs: {data && data.x.toString()}</Text>
      <Text>outputs: {data && data.y.toString()}</Text>
      <Text>predicts: {prediction && prediction.toString()}</Text>
      <Text>{trained ? 'Done Training' : 'Not Yet'}</Text>
      <StatusBar style='auto' />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {},
});
