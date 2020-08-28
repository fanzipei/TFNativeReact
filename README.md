# TFNativeReact

Change the file

TFReactNative\node_modules\@tensorflow\tfjs\node_modules\@tensorflow\tfjs-core\dist\tf-core.node.js

```
function now() {
    //return env().platform.now();
    return 0
}
```

This function only effects the time estimation of deep learning training. If you need this function, you may change it to a correct implementation which can run on Android/iPhone.
