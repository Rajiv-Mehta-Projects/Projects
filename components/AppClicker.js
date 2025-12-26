import React, { useState } from "react";
import {View, StyleSheet,Text, Modal,Button,FlatList,TouchableWithoutFeedback} from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";

import AppClickerItem from "./AppClickerItem";
import AppColors from "../config/AppColors";

// Use Modal to create a drop down menu for catgeories 
function AppClicker({ data, icon, placeholder, onSelectItem, selectedItem }) {
  const [modalVisible, setModalVisible] = useState(false);
  console.log(data);
  return (
    <>
      <TouchableWithoutFeedback onPress={() => setModalVisible(true)}>
        <View style={styles.container}>
          <MaterialCommunityIcons name={icon} size={24} />
          <Text> {selectedItem ? selectedItem.label : placeholder} </Text>
          <MaterialCommunityIcons name="chevron-down" size={24} />
        </View>
      </TouchableWithoutFeedback>
      <Modal visible={modalVisible} animationType="slide">
        <Button title="Close" onPress={() => setModalVisible(false)} />
        <FlatList
          data={data}
          keyExtractor={(item) => item.value.toString()}
          renderItem={({ item }) => (
            <AppClickerItem
              label={item.label}
              onPress={() => {
                setModalVisible(false);
                onSelectItem(item);
              }}
            />
          )}
        />
      </Modal>
    </>
  );
}
const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "center",
    backgroundColor: AppColors.white,
    flexDirection: "row",
    borderRadius: 25,
    padding: 5,
    marginVertical: 15,
    width: "75%",
  },
  textInput: {
    flex: 1,
    fontSize: 15,
    borderRadius: 15,
    width: "50%",
    marginVertical: 10,
    marginLeft: 10,
  },
});

export default AppClicker;
