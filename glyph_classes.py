import tkinter as tk
from abc import abstractmethod


class Glyph:

    @abstractmethod
    def draw(self, begin_column, begin_row):
        pass


class Composition(Glyph):

    def __init__(self, glyphs_list):
        self.glyphs_list = glyphs_list

    @abstractmethod
    def draw(self, begin_column, begin_row):
        pass

    def get_user_information(self):
        user_information = {}
        for current_glyph in self.glyphs_list:
            try:
                user_information.update(current_glyph.get_user_information())
            except AttributeError:
                continue

        return user_information


class Vertical_composition(Composition):

    def __init__(self, glyphs_list):
        Composition.__init__(self, glyphs_list)

    def draw(self, begin_column, begin_row):
        next_row = begin_row
        for current_glyph in self.glyphs_list:
            _, next_row = current_glyph.draw(begin_column, next_row)
        next_column = begin_column + 1
        return next_column, next_row


class Horizontal_composition(Composition):

    def __init__(self, glyphs_list):
        Composition.__init__(self, glyphs_list)

    def draw(self, begin_column, begin_row):
        next_column = begin_column
        for current_glyph in self.glyphs_list:
            next_column, _ = current_glyph.draw(next_column, begin_row)
        next_row = begin_row + 1
        return next_column, next_row


class Button_glyph(Glyph):
    def __init__(self, tk_widget):
        self.tk_widget = tk_widget

    def draw(self, begin_column, begin_row):
        self.tk_widget.grid(column=begin_column, row=begin_row)
        next_column, next_row = begin_column + 1, begin_row + 1
        return next_column, next_row

    def get_user_information(self):
        return {self.tk_widget["text"]: self.tk_widget.get()}


class Radio_button_composition(Horizontal_composition):

    def __init__(self, glyphs_list, button_variable):
        Horizontal_composition.__init__(self, glyphs_list)
        self.button_variable = button_variable

    def get_user_information(self):
        return self.button_variable.get()


class Check_button_composition(Horizontal_composition):

    def __init__(self, glyphs_list, button_variables):
        Horizontal_composition.__init__(self, glyphs_list)
        self.button_variables = button_variables

    def get_user_information(self):
        return {current_variable[0]: current_variable[1].get() for current_variable in self.button_variables.items()}


class Text_field_glyph(Button_glyph):
    def __init__(self, field):
        Button_glyph.__init__(self, field)

    def get_user_information(self):
        return self.tk_widget.get()


class Text_glyph(Glyph):

    def __init__(self, text_label):
        self.text_label = text_label

    def draw(self, begin_column, begin_row):
        self.text_label.grid(column=begin_column, row=begin_row)
        next_column, next_row = begin_column + 1, begin_row + 1
        return next_column, next_row


class Named_Glyph(Glyph):                                  # todo maybe unused
    def __init__(self, glyph_name, glyph_value):
        self.glyph_name = glyph_name
        self.glyph_value = glyph_value

    def draw(self, begin_column, begin_row):
        _, next_row = self.glyph_name.draw(begin_column, begin_row)
        next_column, next_row = self.glyph_value.draw(begin_column, next_row)
        return next_column, next_row

    def get_user_information(self):
        return {self.glyph_name.text_label['text']: self.glyph_value.get_user_information()}


class Named_composition(Glyph):

    def __init__(self, composition_name, composition_value):
        self.composition_name = composition_name
        self.composition_value = composition_value

    def draw(self, begin_column, begin_row):
        _, next_row = self.composition_name.draw(begin_column, begin_row)
        next_column, next_row = self.composition_value.draw(begin_column, next_row)
        return next_column, next_row + 1

    def get_user_information(self):
        return {self.composition_name.text_label['text']: self.composition_value.get_user_information()}


