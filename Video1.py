from manimlib.imports import *
import numpy as np


class s1(Scene):
    def construct(self):
        """This scene is simply giving an episode overview
        -Brief overview of what a neural network is
        -The equations of backpropagation and a particular interpretation of what they mean. This does not mean it is truth it is simply a mode of understanding
        -Have a look at some similar equations
        -Show how neural networks learn using a geometrical approach
        -Show that for a single activation set we get a plane, averaging over all we get a surface and no dependence in activation
        """
        main_title = TextMobject("Geometric Intuition for Neural Networks").scale(1.5).shift(0.5*UP)
        sub_title = TextMobject("Episode 1").scale(1).next_to(main_title, DOWN)
        self.play(FadeIn(main_title))
        self.play(ShowCreation(sub_title))
        self.wait(3)
        self.play(FadeOut(main_title), FadeOut(sub_title))

        video_plan = TextMobject("Aims For This Video:").scale(1).shift(3.5*UP + 4*LEFT)
        vp = []
        vp.append(TextMobject("1) Brief Look At an Artificial Neuron and Network").scale(0.8).next_to(video_plan, 2*DOWN, aligned_edge=LEFT))
        vp.append(TextMobject("2) Similarities Between Back-propagation Equations and Other Formula").scale(0.8).next_to(vp[-1], 2*DOWN, aligned_edge=LEFT))
        vp.append(TextMobject("3) Encounter Weight-Bias-Activation-Cost Space").scale(0.8).next_to(vp[-1], 2*DOWN, aligned_edge=LEFT))
        vp.append(TextMobject("4) Separate into Weight-Bias-Cost Space and Activation Space").scale(0.8).next_to(vp[-1], 2*DOWN, aligned_edge=LEFT))
        vp.append(TextMobject("5) Overview of What We Wish To Achieve Using Neural Networks").scale(0.8).next_to(vp[-1], 2*DOWN, aligned_edge=LEFT))
        vp.append(TextMobject("6) Investigate Some Equations of Motion for Weight-Bias-Cost Space").scale(0.8).next_to(vp[-1], 2*DOWN, aligned_edge=LEFT))
        self.play(FadeIn(video_plan))
        self.wait(1)
        for i in vp:
            self.play(FadeIn(i))
            self.wait(1)
        self.play(ApplyMethod(vp[0].set_color, BLUE), *[FadeOut(i) for i in vp[1:]], FadeOut(video_plan))
        self.play(ApplyMethod(vp[0].move_to, 3*UP))
        self.wait()


class s2(Scene):
    def construct(self):
        vp = TextMobject("1) Brief Look At an Artificial Neuron and Network").scale(0.8).move_to(3*UP).set_color(BLUE)
        self.add(vp)
        self.wait()
        neuron1 = Circle().set_color(RED).scale(0.6)
        connection = []
        connection.append(Line().set_color(WHITE).next_to(neuron1, LEFT))
        connection.append(Line().set_color(WHITE).rotate(45*DEGREES).next_to(neuron1, LEFT))
        connection.append(Line().set_color(WHITE).rotate(-45*DEGREES).next_to(neuron1, LEFT))
        self.play(ShowCreation(neuron1), *[ShowCreation(c) for c in connection])
        self.wait()

class s3(GraphScene):
    CONFIG = {
        "x_min": -5,
        "x_max": 5,
        "y_min": -4,
        "y_max": 4,
        "graph_origin": ORIGIN,
        "function_color": WHITE,
        "axes_color": BLUE,
        "x_axis_label": "$a_0$",
        "y_axis_label": "$a_1$"
    }
    def construct(self):
        #Make graph
        self.setup_axes(animate=True)
        self.w = 1
        self.b = 2
        ymc = TexMobject("y=m\\times x+c").move_to(3*RIGHT+2*UP)
        awb = TexMobject("a_1=w\\times a_0+b").move_to(3*RIGHT+2*UP)
        func_graph = self.get_graph(self.func_to_graph, self.function_color)

        points = []
        numPoints = 40
        minxPoints = -5
        maxxPoints = 5
        for i in range(numPoints):
            x = np.random.randint(maxxPoints - minxPoints) + minxPoints + np.random.rand()
            points.append(SmallDot(self.coords_to_point(x, self.second_line(x) + np.random.normal()/5)))

        #Display graphs

        regression_bias = TexMobject("b = \\frac{\\left(\\sum{a_1}\\right)\\left(\\sum{a_0^2}\\right)-\\left(\\sum{a_0}\\right)\\left(\\sum{a_0a_1}\\right)}{n\\left(\\sum{a_0^2}\\right)-\\left(\\sum{a_0}\\right)^2}").scale(0.5).move_to(4.5*LEFT+3*UP)
        regression_weight = TexMobject("w = \\frac{n\\left(\\sum{a_0a_1}\\right)-\\left(\\sum{a_0}\\right)\\left(\\sum{a_1}\\right)}{n\\left(\\sum{a_0^2}\\right)-\\left(\\sum{a_0}\\right)^2}").scale(0.5).next_to(regression_bias, DOWN, aligned_edge=LEFT)
        point = Dot(self.coords_to_point(1, 1))
       


        #Display graphs
        self.play(Write(ymc), ShowCreation(func_graph))
        self.play(Write(regression_bias), Write(regression_weight))
        for i in range(numPoints):
            self.play(ShowCreation(points[i]), run_time=0.01)
        self.wait(1)
        self.play(ReplacementTransform(ymc, awb))
        self.wait(1)

    def func_to_graph(self, x):
        return self.w*x+self.b
    def second_line(self, x):
        return 0.5*x + 0.25

class s4(Scene):
    def construct(self):
        title = TextMobject("Backpropagation", " Equations").scale(1.5)
        title2 = TextMobject("'Learning'", " Equations").scale(1.5)
        self.play(Write(title))
        self.wait(1)
        self.play(Transform(title, title2))
        self.play(title.move_to, 3*UP, title.scale, 0.8)
        self.wait(1)
        tex = []
        tex.append(TexMobject(r"\delta^{L} = ",r"\nabla",r"_{a}",r" C ",r"\odot",r"\sigma'\left(z^{L}\right)").scale(1).next_to(title, DOWN))
        tex.append(TexMobject(r"\delta^{l} = \left(\left(",r"w",r"^{l+1}\right)^{T}\delta^{l+1}\right)",r"\odot",r"\sigma'\left(z^{l}\right)").scale(1).next_to(tex[-1], DOWN))
        tex.append(TexMobject(r"{\partial ",r"C",r"\over\partial ",r"b",r"_j^{l}} = \delta_j^{l}").scale(1).next_to(tex[-1], DOWN))
        tex.append(TexMobject(r"{\partial ",r"C",r"\over\partial ",r"w",r"_{j k}^{l}} = ",r"a",r"_k^{l-1} \delta_j^{l}").scale(1).next_to(tex[-1], DOWN))
        for i in tex:
            self.play(Write(i))
        self.wait(1)
        # #Highlight a
        self.play(tex[0][2].set_color, RED, tex[3][5].set_color, RED)
        self.wait(0.5)
        self.play(tex[0][2].set_color, WHITE, tex[3][5].set_color, WHITE)
        self.wait(0.5)
        # #Highlight w
        self.play(tex[1][1].set_color, RED, tex[3][3].set_color, RED)
        self.wait(0.5)
        self.play(tex[1][1].set_color, WHITE, tex[3][3].set_color, WHITE)
        self.wait(0.5)
        # #Highlight b
        self.play(tex[2][3].set_color, RED)
        self.wait(0.5)
        self.play(tex[2][3].set_color, WHITE)
        self.wait(0.5)
        # #Highlight c
        self.play(tex[0][3].set_color, RED, tex[2][1].set_color, RED, tex[3][1].set_color, RED)
        self.wait(0.5)
        self.play(tex[0][3].set_color, WHITE, tex[2][1].set_color, WHITE, tex[3][1].set_color, WHITE)
        self.wait(3)
        self.play(tex[0][1].set_color, BLUE, tex[0][4].set_color, YELLOW, tex[1][3].set_color, YELLOW)
        self.wait(1)

        nablaText = TextMobject("Nabla").shift(3*LEFT+1*UP)
        nablaTex = TexMobject("\\nabla").next_to(nablaText, 2*DOWN).scale(3)
        odotText = TextMobject("Elementwise Multiplication").shift(3*RIGHT+1*UP)
        odotTex = TexMobject("\\odot").next_to(odotText, 2*DOWN).scale(3)
        self.play(ReplacementTransform(tex[0][1], nablaTex), ReplacementTransform(tex[0][4], odotTex), *[FadeOut(i) for i in tex[1:]], FadeOut(title))
        self.play(Write(nablaText), Write(odotText), FadeOut(tex[0]))
        self.wait(2)
        self.play(FadeOut(odotText), FadeOut(odotTex))
        self.play(nablaText.shift, 3*RIGHT, nablaTex.shift, 3*RIGHT)
        self.wait(1)

        # #Maxwell's Equations
        me = TextMobject("Maxwell's Equations").scale(0.8).shift(3 * UP + 4 * RIGHT)
        me1 = TexMobject("\\nabla", " \\cdot ","\\bold{E}"," = \\frac{\\rho}{\\epsilon_{0}}").scale(0.5).next_to(me, DOWN, aligned_edge=RIGHT)
        me2 = TexMobject("\\nabla", " \\cdot \\bold{B} = 0").scale(0.5).next_to(me1, DOWN, aligned_edge=RIGHT)
        me3 = TexMobject("\\nabla", " \\times\\bold{E} = -\\frac{\\partial \\bold{B}}{\\partial t}").scale(0.5).next_to(me2, DOWN, aligned_edge=RIGHT)
        me4 = TexMobject("\\nabla", " \\times\\bold{B} = \\mu_{0}\\bold{j}+\\mu_{0}\\epsilon_{0}\\frac{\\partial \\bold{E}}{\\partial t}").scale(0.5).next_to(me3, DOWN, aligned_edge=RIGHT)
        self.play(Write(me))
        self.play(ReplacementTransform(nablaTex[0].copy(), me1[0]))
        self.play(ReplacementTransform(nablaTex[0].copy(), me2[0]), Write(me1[1:]))
        self.play(ReplacementTransform(nablaTex[0].copy(), me3[0]), Write(me2[1:]))
        self.play(ReplacementTransform(nablaTex[0].copy(), me4[0]), Write(me3[1:]))
        self.play(Write(me4[1:]))
        self.wait(1)

        # #Schrödinger Equations
        se = TextMobject(r"Schrodinger Equation").scale(0.8).shift(3 * UP + 4 * LEFT)
        se1 = TexMobject("\\left(\\frac{-\\hbar}{2m}","\\nabla", "+ V\\left(\\bold{r}\\right)\\right)\\bold{\Psi} =i\\hbar\\frac{\\partial\\bold{\\Psi}}{\\partial t}").scale(0.5).next_to(se, DOWN, aligned_edge=LEFT)
        self.play(Write(se))
        self.play(ReplacementTransform(nablaTex[0].copy(), se1[1]))
        self.play(Write(se1[0]), Write(se1[2:]))
        self.wait()

        # #Wave Equations
        we = TextMobject(r"Wave Equation").scale(0.8).next_to(se, DOWN, aligned_edge=LEFT).shift(4*DOWN)
        we1 = TexMobject("\\nabla","^2\\phi= \\frac{1}{c^2}\\frac{\\partial^2\\phi}{\\partial t^2}").scale(0.5).next_to(we, DOWN, aligned_edge=LEFT)
        self.play(Write(we))
        self.play(ReplacementTransform(nablaTex[0].copy(), we1[0]))
        self.play(Write(we1[1:]))
        self.wait()

        # #Diffusion Equations
        de = TextMobject(r"Diffusion Equation").scale(0.8).next_to(me, DOWN, aligned_edge=RIGHT).shift(4*DOWN)
        de1 = TexMobject("\\nabla", "^2T= \\frac{1}{\\alpha^2}\\frac{\\partial T}{\\partial t}}").scale(0.5).next_to(de, DOWN, aligned_edge=RIGHT)
        self.play(Write(de))
        self.play(ReplacementTransform(nablaTex[0].copy(), de1[0]))
        self.play(Write(de1[1:]))
        self.wait(3)


        # #Fade Out equations and show how maxwell's equations describe electric field.
        self.play(*[FadeOut(i) for i in [de, de1, we, we1, se, se1, me3, me4, me2, me, nablaTex, nablaText]], me1.move_to, 0, me1.scale, 3)
        self.wait(1)
        BP = TexMobject(r"\delta^{L} = ",r"\nabla",r"_{a}",r" C ",r"\odot",r"\sigma'\left(z^{L}\right)").scale(1.5).move_to(DOWN)
        self.play(me1.move_to, UP, Write(BP))
        self.play(FadeOut(BP[0]), FadeOut(BP[4:]), FadeOut(BP[4:]), FadeOut(me1[3:]), BP[1:4].move_to, DOWN, me1[:3].move_to, UP)
        self.wait(1)
        self.play(FadeOut(BP[1:3]), FadeOut(me1[:2]))
        electric_field_text = TextMobject("Electric Field").move_to(0.5*UP+2*LEFT)
        cost_field_text = TextMobject("Cost Field","?").move_to(0.5*UP+2*RIGHT)
        self.play(BP[3].next_to, cost_field_text, DOWN,  me1[2].next_to, electric_field_text, DOWN)
        self.play(Write(electric_field_text))
        self.wait(1)
        self.play(Write(cost_field_text))
        self.wait(1)
        self.play(cost_field_text.shift, 2*LEFT, BP[3].shift, 2*LEFT,  FadeOut(me1[2]), FadeOut(electric_field_text), FadeOut(cost_field_text[1]))
        self.wait(1)

