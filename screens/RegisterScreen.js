import React from "react";
import { StyleSheet, View, Image, TouchableOpacity } from "react-native";
import { Formik } from "formik";
import * as Yup from "yup";
import { useNavigation } from "@react-navigation/native";
import * as ImagePicker from "expo-image-picker";

import AppScreen from "../components/AppScreen";
import AppTextInput from "../components/AppTextInput";
import AppButton from "../components/AppButton";
import AppError from "../components/AppError";
import DataManager from "../config/DataManager";

//Sets requirements for email and password and username
const schema = Yup.object().shape({
  email: Yup.string().required().email().label("Email"),
  password: Yup.string().min(4).max(8).label("Password").required(),
  username: Yup.string().required().label("Username"),
});
const getAllUsers = () => {
  let commonData = DataManager.getInstance();
  return commonData.getAllUsers();
};

// Use Formik to create a form and store new user object in memory
function RegisterScreen({ navigation }) {
  const addUser = (values) => {
    let commonData = DataManager.getInstance();
    const user = getAllUsers();
    const userID = user.length + 1;
    const newUser = {
      id: userID,
      username: values.username,
      email: values.email,
      password: values.password,
      pic: values.pic,
    };

    return commonData.addUser(newUser);
  };
  let openImagePickerAsync = async (setField, pic) => {
    let permissionResult =
      await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      alert("Permission to access camera roll is required!");
    }

    let pickerResult = await ImagePicker.launchImageLibraryAsync();

    if (!pickerResult.cancelled) {
      setField(pic, { path: pickerResult.uri });
    }
  };
  return (
    <AppScreen>
      <Formik
        initialValues={{ username: "", email: "", password: "", pic: "" }}
        onSubmit={(values) => {
          addUser(values);
          navigation.navigate("Welcome");
        }}
        validationSchema={schema}
      >
        {({
          values,
          handleChange,
          handleSubmit,
          errors,
          setFieldTouched,
          touched,
          setFieldValue,
        }) => (
          <>
            <View>
              <View>
                <AppTextInput
                  autoCapatilize="none"
                  autoCorrect={false}
                  value={values.username}
                  icon="account-star"
                  placeholder="Username"
                  onBlur={() => setFieldTouched("username")}
                  onChangeText={handleChange("username")}
                />
                {touched.username && <AppError>{errors.username}</AppError>}
                <AppTextInput
                  autoCapatilize="none"
                  autoCorrect={false}
                  value={values.email}
                  icon="email"
                  placeholder="Email Address"
                  keyboardType="email-address"
                  onBlur={() => setFieldTouched("email")}
                  onChangeText={handleChange("email")}
                />
                {touched.email && <AppError> {errors.email}</AppError>}

                <AppTextInput
                  autoCapatilize="none"
                  autoCorrect={false}
                  value={values.password}
                  icon="lock"
                  placeholder="Password"
                  secureTextEntry
                  onBlur={() => setFieldTouched("password")}
                  textContentType="password"
                  onChangeText={handleChange("password")}
                />
                {touched.password && <AppError> {errors.password}</AppError>}
              </View>
              <View>
                {values.pic ? (
                  <TouchableOpacity
                    onPress={() => openImagePickerAsync(setFieldValue, "pic")}
                  >
                    <View
                      style={{
                        alignItems: "center",
                        justifyContent: "space-evenly",
                      }}
                    >
                      <Image
                        source={{ uri: values.pic.path }}
                        style={{
                          height: 200,
                          width: 200,
                          resizeMode: "contain",
                          borderRadius: 500,
                        }}
                      />
                    </View>
                  </TouchableOpacity>
                ) : (
                  <View>
                    <AppButton
                      title={"profile picture"}
                      onPress={() => openImagePickerAsync(setFieldValue, "pic")}
                    />
                  </View>
                )}
                <AppButton title="Submit" onPress={handleSubmit} />
              </View>
            </View>
          </>
        )}
      </Formik>
    </AppScreen>
  );
}
const styles = StyleSheet.create({
  textInputContainer: {
    marginTop: 50,
  },
});

export default RegisterScreen;
