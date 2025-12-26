import React from 'react';
import {Text, TouchableOpacity, StyleSheet,View} from 'react-native';

import AppColors from '../config/AppColors';

function AppButton({title, onPress}) {
    return (
        <TouchableOpacity onPress = {onPress}>
            <View style = {styles.button}> 
            <Text> {title} </Text>
            </View> 
        </TouchableOpacity>
    );
}
const styles = StyleSheet.create({
    button: {
        backgroundColor: AppColors.primaryColor,
        width: '35%',
        alignItems: 'center',
        justifyContent: 'space-between',
        alignSelf: 'center',
        borderRadius: 20,
        padding: 15,

    }
})

export default AppButton;