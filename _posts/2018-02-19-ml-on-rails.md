---
title: ML on Rails
layout: single
author_profile: false
mathjax: true
---

### Learn to build a simple Rails web application that calls a machine learning model and exposes the result with a click of a button.

[_Link to Git_.](https://github.com/TMorville/ml-on-rails)

---

### In this note I will show how to ...

1. Deploy a simple Rails web application using `rails new` and `rails g scaffold` .
2. Use `rest-client`  to send a HTTP request to a machine learning model running in a Docker container.
3. Use `jquery-rails` to make a button that sends a payload to the model, without having to reload the page.

In a upcoming note I will cover HTML and (S)CSS basics for styling a simple web application that scales for different devices. This tutorial does not cover how to build, dockerize or expose a machine learning model via. a RESTful API. If you Google those terms, you will find plenty of tutorials covering this. 

### Prerequisites

* [Docker](https://docs.docker.com/install/)
* [Rails](https://kolosek.com/install-ruby-on-rails-on-ubuntu/)
* [Git](https://gist.github.com/derhuerst/1b15ff4652a867391f03)

### Getting a model

Five seconds of Googling gave me [this](https://github.com/Soluto/python-flask-sklearn-docker-template), which will work just fine. Clone it, and follow the instructions for running on local. When you got it up and running, you can send feature values in the API by changing `f1`, `f2` and `f3` values in the URL. E.g. `http://0.0.0.0:3000/prediction/api/v1.0/some_prediction?f1=4&f2=4&f3=10` should yield `[-1.25]` in your browser. 

### Starting a Rails application

Start our new Rails application:

```bash
rails new ml-on-rails
```

which spawns an entirely new Rails project called `ml-on-rails`. 

We need to add two key components to your `Gemfile`, `'jquery-rails'` that offers higher level of JavaScript methods and `'rest-client'` that allows for communication with our RESTful interface.

```ruby
gem 'jquery-rails'
gem 'rest-client'
```

 Add `//= require jquery3` to `app/assets/javascripts/application.js` as well, and remember to run `bundle install`.

Next, use Rails generator to make a scaffold the controller and view for your application that I am going to call "request". In Rails, [naming matters](https://gist.github.com/iangreenleaf/b206d09c587e8fc6399e), so think about the name of your app for a bit before choosing anything too exotic.

```bash
rails g scaffold request
```

This generates a bunch of files that you don't have to worry too much about at this point. Most notably, your controller now resides in ` /app/controllers/requests_controller.rb` and likewise the files associated with your view in `/app/requests`.

There are three main components we need to configure to make this work: The **controller**, the **view** and the **routes**. If you're in doubt about what these do, find a [relevant blog](https://medium.freecodecamp.org/understanding-the-basics-of-ruby-on-rails-http-mvc-and-routes-359b8d809c7a) and read about the basic MVC architecture that Rails relies on.

### The Controller

When you open your `requests_controller.rb` you're going to see a lot of code that might initially confuse you. These are basic methods that your controller is deployed with and for this note all of them but one are redundant, so go ahead and replace the content of `requests_controller.rb` with the following code. 

```ruby
class RequestsController < ApplicationController

  def show
    f1 = params['f1'].capitalize
    f2 = params['f2'].capitalize
    f3 = params['f3'].capitalize

    response = RestClient::Request.execute(
      method:  :get,
      url:     "http://0.0.0.0:3000/prediction/api/v1.0/some_prediction?f1=#{f1}&f2=#{f2}&f3=#{f3}")

    @result = response

    respond_to do |format|
      format.js {render layout: false}
    end
  end

end
```

Here, I am using the default  `Show` action, but you are allowed to define new actions as well. There are two bits of key code here. First, the code that handles the call to the RESTful interface.

```ruby
response = RestClient::Request.execute(
	method:  :get,
    url:     "http://0.0.0.0:3000/prediction/api/v1.0/some_prediction?f1=#{f1}&f2=#{f2}&f3=#{f3}")
```

Second, the code that formats the JavaScript response and renders the [partial](https://guides.rubyonrails.org/layouts_and_rendering.html#using-partials) in the view that we are going to consider next. 

```ruby
respond_to do |format|
	format.js {render layout: false}
end
```

Essentially, this tells Rails not to use a different layout, because we simply want the result to be displayed on the same page. 

### The View

For the view, I am going to tweak three files. First, `index.html.erb`, that is, the page we are going to use for submitting the request to the model, and getting the response.  

```html
<section id="search">
  <div class="container">
    <%= form_tag("/requests/show", method: "get", remote: true) do %>
      <%= text_field_tag(:f1, "2") %>
      <%= text_field_tag(:f2, "4") %>
      <%= text_field_tag(:f3, "8") %>
      <%= submit_tag("Send request") %>
    <% end %>
  </div>
</section>

<section id="target-for-change">
  <div class="container">
    <% end %>
  </div>
</section>
```

There are two things we should notice in the above. 

First, `form_tag("/requests/show", method: "get", remote: true) do` points to our `show` action defined above. Second, `remote: true` is where the Ajax magic is initialised. In Rails, Ajax combines making requests to a server and updating the information on the page. You can read more about Ajax [here](https://guides.rubyonrails.org/working_with_javascript_in_rails.html#an-introduction-to-ajax).

Second, in `<section id="target-for-change">` we are defining the container where our results should be displayed. This needs to have the same `id` as in `show.html.erb` shown below, with one important change:  `show.html.erb` needs to be renamed to  `show.js.erb` and contains the JavaScript you're interested in showing.

```javascript
$("#target-for-change").html("#{j render(partial: 'show')}");
```

Lastly, we need to define the partial that contains the html we want to update the `target-for-change` with. 

```html
<div class="container">
  Estimate: #{@result}
</div>
```

### The Routes

Standing in your project directory, write `rake routes`. This returns all the exisiting routes in your application and you should see this:

```reStructuredText
requests GET    /requests(.:format)											requests#index                                     
```

which shows your routes and their take parameters. As we only have one page in this application, we want this to be the first thing that meets the user. This is done by adding `root 'requests#index'` to `/config/routes.rb`. 

```ruby
Rails.application.routes.draw do
  root 'requests#index'
  resources :requests
end
```

Notice how `resources :requests` is already added. This happened automatically when we generated the scaffold for the requests controller and adds _all_ the routes added to the requests controller. 

### Running the application

Fire up your new app with `rails s -p 3001` and go to `localhost:3001` . Here you should see three simple input fields, and a `Send request` button. Put in some values, click the button and enjoy the magic. 
