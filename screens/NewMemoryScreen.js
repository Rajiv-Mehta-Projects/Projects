import React, { useState } from "react";
import { TouchableOpacity, View, Image, Text } from "react-native";
import * as ImagePicker from "expo-image-picker";


import AppButton from "../components/AppButton";
import AppClicker from "../components/AppClicker";
import AppColors from "../config/AppColors";
import AppText from "../components/AppText";
import AppScreen from "../components/AppScreen";
import AppTextInput from "../components/AppTextInput";
import DataManager from "../config/DataManager";



function NewMemoryScreen({ navigation }) {
  const [title, setTitle] = useState("");
  const [categoryPicker, setCategoryPicker] = useState("");
  const [image, setImage] = useState(null);
  let commonData = DataManager.getInstance();
  let categories = commonData.getCategory();


// Checks permission for app to access photo library on phone
  let openImagePickerAsync = async () => {
    let permissionResult =
      await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      alert("Permission to access camera roll is required!");
    }

    let pickerResult = await ImagePicker.launchImageLibraryAsync();

    if (pickerResult.cancelled === true) {
      return;
    }
    setImage({ path: pickerResult.uri });
  };

  const addMemory = () => {
    let user = commonData.getUserId();
    const memory = commonData.getMemory(user);
    const memoryID = memory.length + 1;
    const newMemory = {
      title: title,
      category: categoryPicker.label,
      userId: user,
      memoryID: memoryID,
      image: image.path,
    };

    commonData.addMemory(newMemory);
  };

  return (
    <AppScreen>
      <AppText>New Memory</AppText>
      <View style={{ marginTop: 50 }}>
        <AppClicker
          data={categories}
          icon="apps"
          placeholder="Categories"
          selectedItem={categoryPicker}
          onSelectItem={(item) => setCategoryPicker(item)}
        />
      </View>
      <View style={{ marginTop: 10 }}>
        <AppTextInput
          icon="memory"
          placeholder="TITLE"
          value={title}
          onChangeText={(inputText) => setTitle(inputText)}
        />
      </View>
      <View
        style={{
          marginTop: 100,
        }}
      >
        <TouchableOpacity
          onPress={openImagePickerAsync}
          style={{
            alignSelf: "center",
            backgroundColor: AppColors.white,
          }}
        >
          <Text
            style={{
              fontSize: 20,
              alignSelf: "center",
              backgroundColor: AppColors.white,
            }}
          >
            Upload Photo
          </Text>
          {image && (
            <Image
            source={{ uri: image.path }}
              style={{ height: 200, width: 200, resizeMode: "contain" }} 
            />
          )}
        </TouchableOpacity>
      </View>
      <View style={{ marginTop: 100 }}>
        <AppButton
          title="Submit"
          color={AppColors.primaryColor}
          onPress={() => {
            addMemory();
            navigation.navigate("Main");
          }}
        />
      </View>
    </AppScreen>
  );
}

export default NewMemoryScreen;
