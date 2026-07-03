#include <jni.h>

#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_ai_gemmaedge_Native_version(JNIEnv* env, jclass) {
    return env->NewStringUTF("GemmaEdge 0.1");
}

