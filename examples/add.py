import conx as cx

net = cx.Network("Add")
net.debug = True
net.add(cx.Layer("one", 1))
net.add(cx.Layer("two", 1))
net.add(cx.AddLayer("add"))
net.add(cx.Layer("hidden", 2))
net.add(cx.Layer("output", 1))

net.connect("one", "add")
net.connect("two", "add")
net.connect("add", "hidden")
net.connect("hidden", "output")

net.compile(error="mse", optimizer="adam")
