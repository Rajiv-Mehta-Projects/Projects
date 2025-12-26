import React from "react";
import { StyleSheet, View, Image } from "react-native";
import { Formik } from "formik";
import * as Yup from "yup";
import { useNavigation } from "@react-navigation/native";

import AppScreen from "../components/AppScreen";
import AppTextInput from "../components/AppTextInput";
import AppButton from "../components/AppButton";
import AppError from "../components/AppError";
import DataManager from "../config/DataManager";

let commonData = DataManager.getInstance();

// Requirments for email and password
const schema = Yup.object().shape({
  email: Yup.string().required().email().label("Email"),
  password: Yup.string().required().min(4).max(8).label("Password"),
});

const getUser = ({ email }) => {
  return commonData.getUser({ email });
};

const activeUser = ({ email }) => {
  let userId = getUser({ email }).id;
  commonData.setUserId(userId);
};

// Use Formik for user login, checks if email and password matches the database, error messages if incorrect
function WelcomeScreen({ navigation }) {
  return (
    <AppScreen>
      <Image
        style={styles.welcomeLogo}
        source={require("../assets/Logo.png")}
      />

      <Formik
        initialValues={{ email: "", password: "" }}
        onSubmit={(values, { resetForm }) => {
          let commonData = DataManager.getInstance();
          if (commonData.validateUser(values)) {
            resetForm();
            activeUser(values);
            navigation.navigate("Main", {
              screen: "Main",
              params: {
                screen: "Main",
                params: {
                  paramEmail: values.email,
                  paramName: getUser(values).username,
                },
              },
            });
          } else {
            resetForm();
            alert("Invalid Login");
          }
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
        }) => (
          <>
            <View style={styles.textInputContainer}>
              <AppTextInput
                autoCapatilize="none"
                autoCorrect={false}
                icon="email"
                placeholder="Email Address"
                keyboardType="email-address"
                onBlur={() => setFieldTouched("email")}
                textContentType="emailAddress"
                value={values.email}
                onChangeText={handleChange("email")}
              />
              {touched.email && <AppError> {errors.email}</AppError>}
              <AppTextInput
                autoCapatilize="none"
                autoCorrect={false}
                icon="lock"
                placeholder="Password"
                secureTextEntry
                textContentType="password"
                value={values.password}
                onChangeText={handleChange("password")}
              />
              {touched.password && <AppError> {errors.password}</AppError>}
            </View>
            <AppButton title="Login" onPress={handleSubmit} />
          </>
        )}
      </Formik>
      <View style={{ marginTop: 30 }}>
        <AppButton
          title="Register"
          onPress={() => navigation.navigate("Register")}
        />
      </View>
    </AppScreen>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginTop: 150,
  },
  welcomeLogo: {
    justifyContent: "center",
    alignItems: "center",
    marginTop: 90,
    width: 400,
    height: 150,
    resizeMode: "contain",
  },
  textInputContainer: {
    marginVertical: 50,
  },
  button: {
    alignItems: "center",
    justifyContent: "space-between",
    flexDirection: "column",
  },
});

export default WelcomeScreen;
