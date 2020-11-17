#!/usr/bin/env python
from manimlib.imports import *

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flag -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)

# #Scene1 Project Mimir - Geebs and Max
# #Scene2 Neural network and show activations, weights and biases, describing how these are variables and can be tuned to make the network operate as intended
# #Scene3 The way these variable are tuned is using an algorithm called backpropagation utilising gradient descent
# #Scene4 Here are the functions: May notice (elementwise multiplication) and (nabla) are unusual symbols to see.
# #Scene5 Elementwise multiplication is currently of no relevence but we see nabla extensively used in maxwells equations for fields
# #Scene7 Could this imply that cost is roughly a field through space (links in with gradient descent)
# #Scene8 Show that this is the case where weights and biases form the dimensions of this space.



class DrawNeuralNet(Scene):
    def construct(self):
        circle = Circle()
        self.play(ShowCreation(circle))


class BackPropAlgorithms(Scene):
    def construct(self):
        # #Initial Uncoloured Backpropagation Equations:
        text = TextMobject("Backpropagation Equations").scale(1.5).shift(2.5*UP)
        text2 = TextMobject("'Learning Equations'").scale(1.5).shift(2.5 * UP)
        tex = []
        tex.append(TexMobject("\\delta^{L} = \\nabla_{a} C \\odot\\sigma'\\left(z^{L}\\right)").scale(1).next_to(text, DOWN))
        tex.append(TexMobject("\\delta^{l} = \\left(\\left(w^{l+1}\\right)^{T}\\delta^{l+1}\\right)","\\odot","\\sigma'\\left(z^{l}\\right)").scale(1).next_to(tex[-1], DOWN))
        tex.append(TexMobject("\\frac{\\partial C}{\\partial b_j^{l}} = \\delta_j^{l}").scale(1).next_to(tex[-1], DOWN))
        tex.append(TexMobject("\\frac{\\partial C}{\\partial w_{j k}^{l}} = a_k^{l-1} \\delta_j^{l}").scale(1).next_to(tex[-1], DOWN))
        # #Colour activations
        tex.append(TexMobject("\\delta^{L} = \\nabla_{","a","} C \\odot","\\sigma'\\left(z^{L}\\right)").scale(1).next_to(text, DOWN))
        tex.append(TexMobject("\\frac{\\partial C}{\\partial w_{j k}^{l}} = ", "a","_k^{l-1} \\delta_j^{l}").scale(1).next_to(tex[2], DOWN))
        tex[4][1].set_color(RED)
        tex[5][1].set_color(RED)
        # #Colour Weights
        tex.append(TexMobject("\\delta^{l} = \\left(\\left(","w","^{l+1}\\right)^{T}\\delta^{l+1}\\right)", "\\odot", "\\sigma'\\left(z^{l}\\right)").scale(1).next_to(tex[0], DOWN))
        tex.append(TexMobject("\\frac{\\partial C}{\\partial ","w","_{j k}^{l}} = a_k^{l-1} \\delta_j^{l}").scale(1).next_to(tex[2], DOWN))
        tex[6][1].set_color(RED)
        tex[7][1].set_color(RED)
        # #Colour Bias
        tex.append(TexMobject("\\frac{\\partial C}{\\partial ","b","_j^{l}} = \\delta_j^{l}").scale(1).next_to(tex[1], DOWN))
        tex[8][1].set_color(RED)
        # #Look for peculiar symbols
        tex.append(TexMobject("\\delta^{L} = ","\\nabla","_{a} C ","\\odot","\\sigma'\\left(z^{L}\\right)").scale(1).next_to(text, DOWN))
        tex.append(TexMobject("\\delta^{l} = \\left(\\left(w^{l+1}\\right)^{T}\\delta^{l+1}\\right)","\\odot","\\sigma'\\left(z^{l}\\right)").scale(1).next_to(tex[0], DOWN))
        tex[9][1].set_color(YELLOW)
        tex[9][3].set_color(BLUE)
        tex[10][1].set_color(BLUE)
        tex.append(TexMobject("\\nabla").scale(3))
        text3 = TexMobject("Nabla").next_to(tex[-1], UP)
        me = TextMobject("Maxwell's Equations").scale(0.8).shift(2.5*UP).shift(1*UP+3*RIGHT)
        me1 = TexMobject("\\nabla"," \\cdot \\bold{E} = \\frac{\\rho}{\\epsilon_{0}}").scale(0.5).next_to(me, DOWN)
        me2 = TexMobject("\\nabla"," \\cdot \\bold{B} = 0").scale(0.5).next_to(me1, DOWN)
        me3 = TexMobject("\\nabla"," \\times\\bold{E} = -\\frac{\\partial \\bold{B}}{\\partial t}").scale(0.5).next_to(me2, DOWN)
        me4 = TexMobject("\\nabla"," \\times\\bold{B} = \\mu_{0}\\bold{j}+\\mu_{0}\\epsilon_{0}\\frac{\\partial \\bold{E}}{\\partial t}").scale(0.5).next_to(me3, DOWN)
        r"""
        Maxwells Equations
        \nabla \cdot \bold{E} = \frac{\rho}{\epsilon_{0}}\\
        \nabla \cdot \bold{B} = 0\\
        \nabla \times\bold{E} = -\frac{\partial \bold{B}}{\partial t}\\
        \nabla \times\bold{B} = \mu_{0}\bold{j}+\mu_{0}\epsilon_{0}\frac{\partial \bold{E}}{\partial t}\\
        Schrodinger Equation
        \left(\frac{-\hbar}{2m}\nabla+V\left(\bold{r}\right)\right )\bold{\Psi} =i\hbar\frac{\partial\bold{\Psi}}{\partial t}
        Wave Equation
        \nabla^2u=\frac{1}{v^2}\frac{\partial^2u}{\partial t^2}
        """

        self.play(Write(text))
        self.wait(1)
        self.play(ReplacementTransform(text, text2))
        for i in range(4):
            self.play(Write(tex[i]))
        self.wait(1)
        self.play(FadeOut(tex[0]), FadeOut(tex[3]), FadeIn(tex[4]), FadeIn(tex[5]))

        self.wait(1)
        self.play(FadeIn(tex[0]), FadeIn(tex[3]), FadeOut(tex[4]), FadeOut(tex[5]))
        self.play(FadeOut(tex[1]), FadeOut(tex[3]), FadeIn(tex[6]), FadeIn(tex[7]))
        self.wait(1)
        self.play(FadeIn(tex[1]), FadeIn(tex[3]), FadeOut(tex[6]), FadeOut(tex[7]))
        self.play(FadeOut(tex[2]), FadeIn(tex[8]))
        self.wait(1)
        self.play(FadeIn(tex[2]), FadeOut(tex[8]))
        self.play(FadeOut(tex[0]), FadeOut(tex[1]), FadeIn(tex[9]), FadeIn(tex[10]))
        self.wait(2)
        self.play(FadeOut(text), FadeOut(text2), Transform(tex[9][1], tex[11]), FadeOut(tex[10]), FadeOut(tex[2]), FadeOut(tex[3]))
        self.play(FadeInFrom(text3), FadeOut(tex[9][0]), FadeOut(tex[9][2:]))
        self.wait(1)
        self.play(FadeIn(me))

        self.play(ShowCreation(me1[0]))
        self.play(ShowCreation(me2[0]), FadeIn(me1[1]))
        self.play(ShowCreation(me3[0]), FadeIn(me2[1]))
        self.play(ShowCreation(me4[0]), FadeIn(me3[1]))
        self.play(FadeIn(me4[1]))
        self.wait(1)



class Plot1(GraphScene):
    CONFIG = {
        "y_max" : 50,
        "y_min" : 0,
        "x_max" : 7,
        "x_min" : 0,
        "y_tick" : 5,
        "axes_color" : BLUE
    }
    
    def construct(self):
        self.setup_axes()
        graph = self.get_graph(lambda x : x**2, color = GREEN, x_min = 0, x_max = 7)
        self.play(ShowCreation(graph), run_time = 2)
        self.wait()