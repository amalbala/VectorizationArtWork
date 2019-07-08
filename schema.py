"""
GraphQL schema for extracting results from a website.
"""
import graphene
import vectorizer


class FileURL(graphene.ObjectType):
    url = graphene.String()


class VectorizeURL(graphene.Mutation):
    class Arguments:
        url = graphene.String()

    ok = graphene.Boolean()
    outputurl = graphene.String()


    def mutate(root, info, url):
        ok = True
        generator = vectorizer.Vectorizer()
        outputurl = generator.processURL(url)
        return VectorizeURL(ok=ok, outputurl=outputurl)


class Query(graphene.ObjectType):
    # this defines a Field `hello` in our Schema with a single Argument `name`
    hello = graphene.String(name=graphene.String(default_value="stranger"))
    goodbye = graphene.String()

    # our Resolver method takes the GraphQL context (root, info) as well as
    # Argument (name) for the Field and returns data for the query Response
    def resolve_hello(root, info, name):
        return f'Hello {name}!'

    def resolve_goodbye(root, info):
        return 'See ya!'


class Mutation(graphene.ObjectType):
    vectorizeURL = VectorizeURL.Field()


schema = graphene.Schema(query=Query, mutation=Mutation)
